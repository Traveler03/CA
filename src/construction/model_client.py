from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from src.clients.cache import JsonRequestCache
from src.construction.config import ModelConfig
from src.construction.io import write_json


def build_chat_payload(
    model_config: ModelConfig,
    messages: list[dict[str, str]],
    *,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Qwen request using the explicitly selected service protocol."""

    if model_config.endpoint.strip("/") == "responses":
        payload: dict[str, Any] = {
            "model": model_config.name,
            "input": messages,
            "max_output_tokens": model_config.max_completion_tokens,
            "store": False,
        }
        if model_config.reasoning_effort is not None:
            payload["reasoning"] = {"effort": model_config.reasoning_effort}
        if response_format is not None:
            payload["text"] = {"format": response_format}
        return payload

    payload = {
        "model": model_config.name,
        "messages": messages,
        "max_tokens": model_config.max_completion_tokens,
    }
    if model_config.reasoning_effort is not None:
        payload["reasoning_effort"] = model_config.reasoning_effort
    if response_format is not None:
        payload["response_format"] = response_format
    return payload


@dataclass
class ModelResponse:
    ok: bool
    content: str
    raw: dict[str, Any]
    status_code: int | None
    latency_s: float
    attempts: int
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    cache_key: str | None = None
    cached: bool = False


class ApplicationModelClient:
    """Cached asynchronous model calls for English source-grounded construction."""

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        base_url: str,
        api_key: str,
        cache_dir: str | Path,
        raw_dir: str | Path,
        timeout_s: float = 120.0,
    ) -> None:
        self.model_config = model_config
        self.base_url = base_url.rstrip("/")
        self.cache = JsonRequestCache(cache_dir, enabled=model_config.cache_enabled)
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        self.network_call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    async def aclose(self) -> None:
        return None

    async def complete_once(
        self,
        messages: list[dict[str, str]],
        *,
        namespace: str,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        payload = build_chat_payload(self.model_config, messages, response_format=response_format)
        cache_key = self.cache.key_for(_safe_namespace(namespace), payload)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return ModelResponse(
                ok=bool(cached.get("ok")),
                content=str(cached.get("content", "")),
                raw=cached.get("raw", {}) if isinstance(cached.get("raw"), dict) else {},
                status_code=cached.get("status_code"),
                latency_s=0.0,
                attempts=0,
                usage=cached.get("usage", {}) if isinstance(cached.get("usage"), dict) else {},
                error=cached.get("error"),
                cache_key=cache_key,
                cached=True,
            )

        async with self._get_semaphore():
            start = time.perf_counter()
            self.network_call_count += 1
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_s), headers=self._headers) as http_client:
                    response = await http_client.post(
                        f"{self.base_url}/{self.model_config.endpoint.lstrip('/')}",
                        json=payload,
                    )
                latency = time.perf_counter() - start
                try:
                    raw = response.json()
                except Exception:
                    raw = {"text": response.text}
                usage = raw.get("usage", {}) if isinstance(raw, dict) and isinstance(raw.get("usage"), dict) else {}
                self._accumulate_usage(usage)
                content = _extract_content(raw)
                result = ModelResponse(
                    ok=200 <= response.status_code < 300 and bool(content) and _completed(raw),
                    content=content,
                    raw=raw if isinstance(raw, dict) else {"raw": raw},
                    status_code=response.status_code,
                    latency_s=latency,
                    attempts=1,
                    usage=usage,
                    error=None if 200 <= response.status_code < 300 and content and _completed(raw)
                    else _safe_error(raw) if _completed(raw) else "incomplete_model_output",
                    cache_key=cache_key,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                result = ModelResponse(
                    ok=False,
                    content="",
                    raw={"exception": type(exc).__name__, "message": str(exc)},
                    status_code=None,
                    latency_s=time.perf_counter() - start,
                    attempts=1,
                    error=type(exc).__name__,
                    cache_key=cache_key,
                )

        self._save_raw(cache_key, result.raw)
        if result.ok:
            self.cache.set(cache_key, result.__dict__)
        else:
            # Keep raw failure records while allowing resume to retry transport failures.
            self.cache.path_for(cache_key).unlink(missing_ok=True)
        return result

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        namespace: str,
        validate: Callable[[dict[str, Any]], Any],
    ) -> tuple[Any | None, ModelResponse | None, str | None]:
        """Call the model and validate a JSON object, using at most max_retries+1 calls."""

        working_messages = list(messages)
        last_result: ModelResponse | None = None
        last_error: str | None = None
        attempts_allowed = self.model_config.max_retries + 1
        for attempt in range(1, attempts_allowed + 1):
            result = await self.complete_once(working_messages, namespace=f"{namespace}.attempt_{attempt}")
            last_result = result
            if not result.ok:
                last_error = result.error or "model_call_failed"
            else:
                try:
                    parsed = parse_json_object(result.content)
                    value = validate(parsed)
                    return value, result, None
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"

            if attempt < attempts_allowed:
                working_messages = [
                    *messages,
                    *([{"role": "assistant", "content": result.content}] if result.ok else []),
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was invalid for this schema. "
                            f"Error: {last_error}. Fix these errors in the previous JSON while preserving valid content. "
                            "Do not change source text or relax the required schema. Return only a corrected JSON object."
                        ),
                    },
                ]
                await asyncio.sleep(_backoff_s(attempt))
        return None, last_result, last_error

    def usage_summary(self) -> dict[str, int]:
        return {
            "network_model_calls": self.network_call_count,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def _accumulate_usage(self, usage: dict[str, Any]) -> None:
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or prompt + completion)
        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
        self.total_tokens += total

    def _get_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self.model_config.concurrency)
            self._semaphore_loop = loop
        return self._semaphore

    def _save_raw(self, key: str, raw: dict[str, Any]) -> None:
        path = self.raw_dir / f"{key}.json"
        # A resumed HTTP retry uses the same cache key. Retain its previous raw failure.
        if path.exists():
            previous = self.raw_dir / f"{key}.previous-{time.time_ns()}.json"
            path.replace(previous)
        write_json(path, raw)


def _safe_namespace(namespace: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", namespace)


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    # Some Qwen deployments put thinking inline instead of in reasoning_content.
    if text.startswith("<think>"):
        _, separator, text = text.partition("</think>")
        if not separator:
            raise ValueError("unfinished thinking block; no final JSON answer")
        text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("JSON root must be an object")
    return parsed


def _extract_content(raw: Any) -> str:
    if not isinstance(raw, dict):
        return ""
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = first.get("text")
            if isinstance(text, str):
                return text
    output_text = raw.get("output_text")
    if isinstance(output_text, str):
        return output_text
    output = raw.get("output")
    if isinstance(output, list):
        return "".join(
            part["text"]
            for item in output if isinstance(item, dict) and item.get("type") == "message"
            for part in item.get("content", [])
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str)
        )
    return ""


def _completed(raw: Any) -> bool:
    if not isinstance(raw, dict) or raw.get("status") in {"incomplete", "failed", "cancelled"}:
        return False
    choices = raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0].get("finish_reason") in (None, "stop")
    return True


def _safe_error(raw: Any) -> str:
    if isinstance(raw, dict):
        error = raw.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or "api_error")
        if error:
            return str(error)
        if raw.get("message"):
            return str(raw["message"])
    return "empty_content"


def _backoff_s(attempt: int) -> float:
    return min(8.0, (2 ** (attempt - 1)) + random.uniform(0.0, 0.25))
