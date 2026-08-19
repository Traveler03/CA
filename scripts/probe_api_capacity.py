from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
import yaml

from src.runtime.resolve_codex_provider import resolve_provider


CHAT_LADDER = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
EMBED_CONCURRENCY_LADDER = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
EMBED_BATCH_LADDER = [1, 8, 16, 32, 64]


@dataclass
class RequestProbe:
    ok: bool
    status_code: int | None
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error_type: str | None = None
    hidden_reasoning: bool = False
    content_empty: bool = False
    finish_reason: str | None = None
    logprobs_exposed: bool = False


@dataclass
class ConcurrencyMetrics:
    concurrency: int
    batch_size: int | None
    requests: int
    successes: int
    success_rate: float
    count_429: int
    count_5xx: int
    timeouts: int
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    requests_per_s: float
    tokens_per_s: float
    texts_per_s: float | None = None


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> RequestProbe:
    start = time.perf_counter()
    try:
        response = await client.post(url, json=payload, timeout=timeout_s)
        latency = time.perf_counter() - start
        try:
            raw = response.json()
        except Exception:
            raw = {}
        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        status = response.status_code
        message = _first_message(raw)
        content = message.get("content") if isinstance(message, dict) else None
        return RequestProbe(
            ok=200 <= status < 300,
            status_code=status,
            latency_s=latency,
            prompt_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            error_type=_classify_status(status, raw),
            hidden_reasoning=_has_hidden_reasoning(raw),
            content_empty=content == "",
            finish_reason=_finish_reason(raw),
            logprobs_exposed=_has_logprobs(raw),
        )
    except httpx.TimeoutException:
        return RequestProbe(False, None, time.perf_counter() - start, error_type="timeout")
    except Exception as exc:
        return RequestProbe(False, None, time.perf_counter() - start, error_type=type(exc).__name__)


async def _get_json(client: httpx.AsyncClient, url: str, timeout_s: float) -> tuple[bool, int | None, dict[str, Any] | str]:
    try:
        response = await client.get(url, timeout=timeout_s)
        try:
            payload: dict[str, Any] | str = response.json()
        except Exception:
            payload = response.text[:500]
        return 200 <= response.status_code < 300, response.status_code, payload
    except Exception as exc:
        return False, None, type(exc).__name__


def _classify_status(status: int, raw: Any) -> str | None:
    if 200 <= status < 300:
        return None
    if status == 429:
        return "429"
    if status >= 500:
        return "5xx"
    if isinstance(raw, dict) and "error" in raw:
        return str(raw["error"])[:200]
    return str(status)


def _first_message(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    choices = raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            return message
    return {}


def _has_hidden_reasoning(raw: Any) -> bool:
    message = _first_message(raw)
    return bool(message.get("reasoning") or message.get("reasoning_content") or message.get("reasoning_details"))


def _finish_reason(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        return str(reason) if reason is not None else None
    return None


def _has_logprobs(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    choices = raw.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if isinstance(choice, dict) and choice.get("logprobs"):
            return True
    return False


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1)
    return ordered[idx]


async def _run_load(
    *,
    concurrency: int,
    n_requests: int,
    batch_size: int | None,
    call: Callable[[], Awaitable[RequestProbe]],
) -> ConcurrencyMetrics:
    semaphore = asyncio.Semaphore(concurrency)
    start = time.perf_counter()

    async def guarded() -> RequestProbe:
        async with semaphore:
            return await call()

    results = await asyncio.gather(*(guarded() for _ in range(n_requests)))
    elapsed = max(time.perf_counter() - start, 1e-9)
    latencies = [r.latency_s for r in results]
    successes = sum(1 for r in results if r.ok)
    total_tokens = sum(r.total_tokens or (r.prompt_tokens + r.completion_tokens) for r in results)
    texts = (batch_size or 1) * len(results)
    return ConcurrencyMetrics(
        concurrency=concurrency,
        batch_size=batch_size,
        requests=len(results),
        successes=successes,
        success_rate=successes / len(results) if results else 0.0,
        count_429=sum(1 for r in results if r.error_type == "429"),
        count_5xx=sum(1 for r in results if r.error_type == "5xx"),
        timeouts=sum(1 for r in results if r.error_type == "timeout"),
        p50_latency_s=statistics.median(latencies) if latencies else 0.0,
        p95_latency_s=_percentile(latencies, 95),
        p99_latency_s=_percentile(latencies, 99),
        requests_per_s=len(results) / elapsed,
        tokens_per_s=total_tokens / elapsed,
        texts_per_s=texts / elapsed if batch_size is not None else None,
    )


def _should_stop(metric: ConcurrencyMetrics, baseline_p95: float) -> bool:
    if 1.0 - metric.success_rate > 0.10:
        return True
    if metric.count_429 >= 2:
        return True
    if metric.count_5xx >= 2:
        return True
    if baseline_p95 > 0 and metric.p95_latency_s > baseline_p95 * 4:
        return True
    return False


def _safe_concurrency(metrics: list[ConcurrencyMetrics], baseline_p95: float) -> int:
    safe = 0
    for metric in metrics:
        if (
            metric.success_rate >= 0.99
            and metric.count_429 == 0
            and metric.timeouts == 0
            and (baseline_p95 == 0 or metric.p95_latency_s <= baseline_p95 * 2.5)
        ):
            safe = metric.concurrency
    return max(1, safe)


async def probe(args: argparse.Namespace) -> dict[str, Any]:
    resolved, api_key = resolve_provider(
        ".",
        chat_model_override=args.chat_model,
        embedding_model_override=args.embedding_model,
    )
    base_url = resolved.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout_s = args.timeout_seconds
    quick = args.quick
    chat_ladder = [1, 2] if quick else CHAT_LADDER
    emb_conc_ladder = [1, 2] if quick else EMBED_CONCURRENCY_LADDER
    emb_batch_ladder = [1, 8] if quick else EMBED_BATCH_LADDER
    requests_per_step = 4 if quick else 32
    confirmation_requests = 8 if quick else 100
    warmups = 1 if quick else 3

    async with httpx.AsyncClient(headers=headers) as client:
        models_ok, models_status, models_payload = await _get_json(client, f"{base_url}/models", timeout_s)
        capabilities: dict[str, Any] = {
            "models_endpoint": {"ok": models_ok, "status_code": models_status},
            "models_sample": _model_ids_sample(models_payload),
        }

        chat_payload = _chat_payload(resolved.chat_model, "Return only OK.", max_tokens=8)
        chat_basic = await _post_json(client, f"{base_url}/chat/completions", chat_payload, timeout_s)
        capabilities["chat_completion"] = asdict(chat_basic)

        json_payload = _chat_payload(
            resolved.chat_model,
            'Return exactly {"answer":"A"} as JSON.',
            max_tokens=20,
            response_format={"type": "json_object"},
        )
        capabilities["json_mode"] = asdict(await _post_json(client, f"{base_url}/chat/completions", json_payload, timeout_s))

        logprob_payload = _chat_payload(
            resolved.chat_model,
            'Answer with exactly "Yes".',
            max_tokens=2,
            logprobs=True,
            top_logprobs=5,
        )
        capabilities["logprobs"] = asdict(await _post_json(client, f"{base_url}/chat/completions", logprob_payload, timeout_s))

        think_payload = _chat_payload(resolved.chat_model, "Return only A.", max_tokens=8)
        capabilities["enable_thinking_false"] = asdict(await _post_json(client, f"{base_url}/chat/completions", think_payload, timeout_s))

        embedding_capabilities: dict[str, Any] = {}
        embedding_basic_payload = {"model": resolved.embedding_model, "input": ["short text"]}
        emb_basic = await _post_json(client, f"{base_url}/embeddings", embedding_basic_payload, timeout_s)
        embedding_capabilities["embedding_endpoint"] = asdict(emb_basic)
        if emb_basic.ok:
            raw = await _embedding_raw(client, f"{base_url}/embeddings", embedding_basic_payload, timeout_s)
            embedding_capabilities["output_dimension"] = _embedding_dimension(raw)
        empty_payload = {"model": resolved.embedding_model, "input": [""]}
        embedding_capabilities["empty_string"] = asdict(await _post_json(client, f"{base_url}/embeddings", empty_payload, timeout_s))
        long_payload = {"model": resolved.embedding_model, "input": ["long " * 4096]}
        embedding_capabilities["long_text"] = asdict(await _post_json(client, f"{base_url}/embeddings", long_payload, timeout_s))

        chat_metrics: list[ConcurrencyMetrics] = []
        if chat_basic.ok:
            for c in chat_ladder:
                for _ in range(warmups):
                    await _post_json(client, f"{base_url}/chat/completions", chat_payload, timeout_s)

                metric = await _run_load(
                    concurrency=c,
                    n_requests=requests_per_step,
                    batch_size=None,
                    call=lambda: _post_json(client, f"{base_url}/chat/completions", chat_payload, timeout_s),
                )
                chat_metrics.append(metric)
                baseline = chat_metrics[0].p95_latency_s if chat_metrics else 0.0
                if _should_stop(metric, baseline):
                    break
            if chat_metrics:
                candidate = _safe_concurrency(chat_metrics, chat_metrics[0].p95_latency_s)
                await _run_load(
                    concurrency=candidate,
                    n_requests=confirmation_requests,
                    batch_size=None,
                    call=lambda: _post_json(client, f"{base_url}/chat/completions", chat_payload, timeout_s),
                )

        embedding_metrics: list[ConcurrencyMetrics] = []
        max_working_batch = 0
        if emb_basic.ok:
            for batch_size in emb_batch_ladder:
                payload = {"model": resolved.embedding_model, "input": [f"text {i}" for i in range(batch_size)]}
                result = await _post_json(client, f"{base_url}/embeddings", payload, timeout_s)
                embedding_capabilities[f"batch_{batch_size}"] = asdict(result)
                if result.ok:
                    max_working_batch = batch_size
            chosen_batch = max_working_batch or 1
            for c in emb_conc_ladder:
                payload = {"model": resolved.embedding_model, "input": [f"text {i}" for i in range(chosen_batch)]}
                metric = await _run_load(
                    concurrency=c,
                    n_requests=requests_per_step,
                    batch_size=chosen_batch,
                    call=lambda payload=payload: _post_json(client, f"{base_url}/embeddings", payload, timeout_s),
                )
                embedding_metrics.append(metric)
                baseline = embedding_metrics[0].p95_latency_s if embedding_metrics else 0.0
                if _should_stop(metric, baseline):
                    break

    chat_safe = _safe_concurrency(chat_metrics, chat_metrics[0].p95_latency_s) if chat_metrics else 1
    emb_safe = _safe_concurrency(embedding_metrics, embedding_metrics[0].p95_latency_s) if embedding_metrics else 1
    runtime_chat = max(1, math.floor(chat_safe * 0.8))
    runtime_emb = max(1, math.floor(emb_safe * 0.8))
    payload = {
        "provider": resolved.public_dict(),
        "capabilities": capabilities,
        "embedding_capabilities": embedding_capabilities,
        "chat_metrics": [asdict(m) for m in chat_metrics],
        "embedding_metrics": [asdict(m) for m in embedding_metrics],
        "max_observed_chat_concurrency": max((m.concurrency for m in chat_metrics), default=0),
        "safe_chat_concurrency": chat_safe,
        "runtime_chat_concurrency": runtime_chat,
        "max_observed_embedding_concurrency": max((m.concurrency for m in embedding_metrics), default=0),
        "safe_embedding_concurrency": emb_safe,
        "runtime_embedding_concurrency": runtime_emb,
        "embedding_batch_size": max_working_batch or 1,
        "quick_mode": quick,
    }
    return payload


def _chat_payload(
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    response_format: dict[str, Any] | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning": {"enabled": False},
        "reasoning_effort": "none",
    }
    if response_format:
        payload["response_format"] = response_format
    if logprobs is not None:
        payload["logprobs"] = logprobs
    if top_logprobs is not None:
        payload["top_logprobs"] = top_logprobs
    return payload


async def _embedding_raw(client: httpx.AsyncClient, url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    try:
        response = await client.post(url, json=payload, timeout=timeout_s)
        return response.json()
    except Exception as exc:
        return {"error": type(exc).__name__}


def _embedding_dimension(raw: dict[str, Any]) -> int | None:
    data = raw.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        emb = data[0].get("embedding")
        if isinstance(emb, list):
            return len(emb)
    return None


def _model_ids_sample(payload: dict[str, Any] | str) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for item in data[:20]:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.append(item["id"])
    return ids


def write_outputs(payload: dict[str, Any]) -> None:
    reports = Path("reports")
    configs = Path("configs")
    reports.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)
    (reports / "api_capacity_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (reports / "API_CAPACITY_REPORT.md").write_text(render_markdown(payload), encoding="utf-8")
    (reports / "api_capacity_report.md").write_text(render_markdown(payload), encoding="utf-8")

    runtime_config = {
        "api": {
            "chat_concurrency": payload["runtime_chat_concurrency"],
            "embedding_concurrency": payload["runtime_embedding_concurrency"],
            "embedding_batch_size": payload["embedding_batch_size"],
            "timeout_seconds": 60,
            "max_retries": 3,
        },
        "models": {
            "chat": payload["provider"]["chat_model"],
            "embedding": payload["provider"]["embedding_model"],
        },
    }
    (configs / "runtime.auto.yaml").write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    provider = payload["provider"]
    lines = [
        "# API capacity report",
        "",
        f"Mode: {'quick sanity' if payload.get('quick_mode') else 'full probe'}",
        "",
        "## Provider",
        "",
        f"- base_url host: `{provider.get('base_url_host', '')}`",
        f"- base_url source: `{provider.get('base_url_source', '')}`",
        f"- key source: `{provider.get('key_source', '')}`",
        f"- masked key: `{provider.get('key_masked', '')}`",
        f"- chat model: `{provider.get('chat_model', '')}`",
        f"- embedding model: `{provider.get('embedding_model', '')}`",
        "",
        "## Capability checks",
        "",
        f"- `/v1/models`: {payload['capabilities']['models_endpoint']}",
        f"- Chat Completion: {payload['capabilities']['chat_completion']}",
        f"- JSON mode: {payload['capabilities']['json_mode']}",
        f"- logprobs: {payload['capabilities']['logprobs']}",
        f"- `chat_template_kwargs.enable_thinking=false`: {payload['capabilities']['enable_thinking_false']}",
        f"- Embedding endpoint: {payload['embedding_capabilities'].get('embedding_endpoint')}",
        f"- Embedding output dimension: {payload['embedding_capabilities'].get('output_dimension')}",
        f"- Empty string behavior: {payload['embedding_capabilities'].get('empty_string')}",
        f"- Long text behavior: {payload['embedding_capabilities'].get('long_text')}",
        "",
        "## Chat concurrency",
        "",
        f"- max_observed_concurrency: {payload['max_observed_chat_concurrency']}",
        f"- safe_concurrency: {payload['safe_chat_concurrency']}",
        f"- runtime_concurrency: {payload['runtime_chat_concurrency']}",
        "",
        "```json",
        json.dumps(payload["chat_metrics"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Embedding concurrency",
        "",
        f"- max_observed_concurrency: {payload['max_observed_embedding_concurrency']}",
        f"- safe_concurrency: {payload['safe_embedding_concurrency']}",
        f"- runtime_concurrency: {payload['runtime_embedding_concurrency']}",
        f"- embedding_batch_size: {payload['embedding_batch_size']}",
        "",
        "```json",
        json.dumps(payload["embedding_metrics"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Notes",
        "",
        "- Safe concurrency is the highest observed level satisfying success-rate, 429, timeout, and p95 constraints.",
        "- Runtime concurrency is `floor(safe_concurrency * 0.8)`, minimum 1.",
        "- Reports intentionally omit raw API keys.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe Chat and Embedding API capacity.")
    parser.add_argument("--quick", action="store_true", help="Run a low-cost sanity probe instead of the full ladder.")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--chat-model")
    parser.add_argument("--embedding-model")
    args = parser.parse_args(argv)
    payload = asyncio.run(probe(args))
    write_outputs(payload)
    print(render_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
