#!/usr/bin/env python3
"""Split a validated JSONL input into deterministic resumable shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("--shards must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.output_dir / f"shard{index}.jsonl" for index in range(args.shards)]
    counts = [0] * args.shards
    handles = [path.open("w", encoding="utf-8") for path in paths]
    try:
        with args.input.open(encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {args.input}:{number}") from exc
                if not row.get("language") or not row.get("sample_id"):
                    raise ValueError(f"Missing language/sample_id at {args.input}:{number}")
                shard = (number - 1) % args.shards
                handles[shard].write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                counts[shard] += 1
    finally:
        for handle in handles:
            handle.close()
    print(json.dumps({"input": str(args.input), "shards": [str(path) for path in paths], "counts": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
