from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests


def batched(rows: list[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def post_batch(endpoint: str, queries: list[str], top_k: int, *, retries: int = 3, timeout_s: float = 600.0) -> dict[str, Any]:
    payload = {"queries": queries, "top_k": top_k}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout_s)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    raise RuntimeError(f"batch request failed after {retries} attempts: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk query the local Sherlock Wikipedia search service.")
    parser.add_argument("--input", required=True, type=Path, help="JSONL input.")
    parser.add_argument("--output", required=True, type=Path, help="JSONL output.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8897/search_batch")
    parser.add_argument("--query-field", default="query")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if args.limit is not None and len(rows) >= args.limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            query = row.get(args.query_field)
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"missing string query field {args.query_field!r} at line {line_no}")
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    started = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as out:
        for batch in batched(rows, args.batch_size):
            queries = [str(row[args.query_field]) for row in batch]
            response = post_batch(args.endpoint, queries, args.top_k)
            results = response.get("results")
            if not isinstance(results, list) or len(results) != len(batch):
                raise RuntimeError(f"unexpected batch response shape: {response.keys()}")
            for row, item in zip(batch, results):
                out.write(
                    json.dumps(
                        {
                            args.id_field: row.get(args.id_field),
                            "query": row[args.query_field],
                            "wiki_results": item.get("results", []),
                            "wiki_latency_s": item.get("latency_s"),
                            "error": item.get("error"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            completed += len(batch)
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(rows),
                        "elapsed_s": round(time.perf_counter() - started, 3),
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
