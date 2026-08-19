#!/usr/bin/env python3
"""Audit per-method Global-MMLU coverage without rewriting benchmark data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


LOW_RESOURCE_LANGUAGES = ("bn", "hi", "ne", "sw", "te")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/global_mmlu"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="METHOD=PATH",
        help="One historical prediction JSONL per method.",
    )
    return parser.parse_args()


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["language"]), str(row["sample_id"])


def split_name(sample_id: str) -> str:
    parts = sample_id.split("/")
    return parts[-2] if len(parts) > 1 else "unknown"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}") from exc


def main() -> int:
    args = parse_args()
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for language in LOW_RESOURCE_LANGUAGES:
        for row in read_jsonl(args.data_dir / f"{language}.jsonl"):
            if row.get("language") != language:
                raise ValueError(f"Language mismatch in {language}.jsonl")
            key = row_key(row)
            if key in expected:
                raise ValueError(f"Duplicate benchmark key: {key}")
            expected[key] = row

    parsed_predictions: dict[str, Path] = {}
    for item in args.prediction:
        method, separator, raw_path = item.partition("=")
        if not separator or not method or not raw_path:
            raise ValueError(f"Expected METHOD=PATH, got {item!r}")
        if method in parsed_predictions:
            raise ValueError(f"Duplicate method: {method}")
        parsed_predictions[method] = Path(raw_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods: dict[str, Any] = {}
    for method, prediction_path in parsed_predictions.items():
        covered: set[tuple[str, str]] = set()
        off_scope = 0
        duplicate_rows = 0
        for row in read_jsonl(prediction_path):
            if row.get("dataset") != "global_mmlu":
                continue
            language = str(row.get("language", ""))
            sample_id = str(row.get("sample_id", ""))
            key = (language, sample_id)
            if key not in expected:
                off_scope += 1
                continue
            if key in covered:
                duplicate_rows += 1
            covered.add(key)
        missing = sorted(set(expected) - covered)
        missing_rows = [expected[key] | {"_dataset": "global_mmlu"} for key in missing]
        missing_path = args.output_dir / f"missing_{method}.jsonl"
        with missing_path.open("w", encoding="utf-8") as handle:
            for row in missing_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        methods[method] = {
            "historical_prediction_path": str(prediction_path),
            "covered_unique": len(covered),
            "missing": len(missing),
            "duplicate_rows": duplicate_rows,
            "off_scope_rows": off_scope,
            "covered_by_language": dict(sorted(Counter(language for language, _ in covered).items())),
            "missing_by_language": dict(sorted(Counter(language for language, _ in missing).items())),
            "covered_by_split": dict(sorted(Counter(split_name(sample_id) for _, sample_id in covered).items())),
            "missing_by_split": dict(sorted(Counter(split_name(sample_id) for _, sample_id in missing).items())),
            "missing_input": str(missing_path),
        }

    report = {
        "scope": {"languages": list(LOW_RESOURCE_LANGUAGES), "expected_unique": len(expected)},
        "methods": methods,
    }
    (args.output_dir / "coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
