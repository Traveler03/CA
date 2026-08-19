from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.smoke_test.config import ModelConfig
from src.smoke_test.model_client import SmokeModelClient


def prompt(i: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Return only compact JSON. No markdown.",
        },
        {
            "role": "user",
            "content": f'Return exactly JSON {{"ok":true,"n":{i},"label":"probe"}}.',
        },
    ]


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("ok") is not True:
        raise ValueError("ok must be true")
    return payload


async def run_level(args: argparse.Namespace, concurrency: int) -> dict[str, Any]:
    call_count = max(args.min_calls, concurrency * args.waves)
    model_config = ModelConfig(
        name=args.model,
        max_completion_tokens=args.max_completion_tokens,
        concurrency=concurrency,
        max_retries=args.max_retries,
        cache_enabled=False,
    )
    out = Path(args.output_dir) / f"c{concurrency}"
    client = SmokeModelClient.from_config(
        model_config,
        cache_dir=out / "cache",
        raw_dir=out / "raw",
        timeout_s=args.timeout_s,
    )
    started = time.perf_counter()
    try:
        tasks = [
            client.complete_json(
                prompt(i),
                namespace=f"gpt54_concurrency.c{concurrency}.{i}.{time.time_ns()}",
                validate=validate,
            )
            for i in range(call_count)
        ]
        outputs = await asyncio.gather(*tasks)
    finally:
        await client.aclose()
    elapsed = time.perf_counter() - started
    latencies = [result.latency_s for _payload, result, _error in outputs if result is not None]
    ok = sum(1 for payload, _result, error in outputs if payload is not None and error is None)
    errors: dict[str, int] = {}
    status_codes: dict[str, int] = {}
    for _payload, result, error in outputs:
        if error:
            errors[error] = errors.get(error, 0) + 1
        if result is not None:
            key = str(result.status_code)
            status_codes[key] = status_codes.get(key, 0) + 1
    return {
        "model": args.model,
        "concurrency": concurrency,
        "call_count": call_count,
        "ok": ok,
        "failed": call_count - ok,
        "success_rate": ok / call_count if call_count else 0.0,
        "elapsed_s": elapsed,
        "throughput_rps": ok / elapsed if elapsed > 0 else 0.0,
        "latency_p50_s": statistics.median(latencies) if latencies else None,
        "latency_p95_s": sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else None,
        "latency_max_s": max(latencies) if latencies else None,
        "status_codes": status_codes,
        "errors": errors,
        "usage": client.usage_summary(),
    }


async def run_probe(args: argparse.Namespace) -> list[dict[str, Any]]:
    results = []
    for concurrency in args.concurrency_levels:
        result = await run_level(args, concurrency)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if result["success_rate"] < args.stop_below_success_rate:
            break
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe GPT-5.4 API concurrency with short JSON calls.")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--waves", type=int, default=2)
    parser.add_argument("--min-calls", type=int, default=2)
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--stop-below-success-rate", type=float, default=0.80)
    parser.add_argument("--output-dir", default="runs/smoke_001/gpt54_concurrency_probe")
    args = parser.parse_args(argv)
    asyncio.run(run_probe(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
