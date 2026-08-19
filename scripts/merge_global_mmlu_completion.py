#!/usr/bin/env python3
"""Merge historical and completion predictions with exact five-language coverage checks."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LANGUAGES = ("bn", "hi", "ne", "sw", "te")


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


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("language") or ""), str(row.get("sample_id") or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/global_mmlu"))
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--completion", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def compact(row: dict[str, Any], expected: dict[str, Any], source: str, method: str) -> dict[str, Any]:
    answer = str(expected["answer"]).strip().upper()
    prediction = row.get("prediction")
    if prediction is not None:
        prediction = str(prediction).strip().upper()
    # Older zero-shot artifacts predate the explicit `valid` field. Their
    # normalized A-D prediction is nevertheless a valid parsed answer.
    valid = bool(row.get("valid")) if "valid" in row else prediction in {"A", "B", "C", "D"}
    return {
        "method": method,
        "dataset": "global_mmlu",
        "language": expected["language"],
        "sample_id": expected["sample_id"],
        "subject": expected.get("subject"),
        "subject_category": expected.get("subject_category"),
        "source_dataset_split": expected.get("source_dataset_split"),
        "answer": answer,
        "prediction": prediction,
        "valid": valid,
        "correct": bool(valid and prediction == answer),
        "result_source": source,
        "latency_s": row.get("latency_s"),
        "error": row.get("error"),
    }


def main() -> int:
    args = parse_args()
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for language in LANGUAGES:
        for row in read_jsonl(args.data_dir / f"{language}.jsonl"):
            item_key = key(row)
            if item_key in expected:
                raise ValueError(f"Duplicate source key: {item_key}")
            expected[item_key] = row

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    sources = [("historical", args.historical), *[("completion", path) for path in args.completion]]
    for source, path in sources:
        for row in read_jsonl(path):
            if str(row.get("dataset") or "global_mmlu") != "global_mmlu":
                continue
            item_key = key(row)
            if item_key not in expected:
                continue
            if item_key in merged:
                raise ValueError(f"Duplicate merged key {item_key} from {path}")
            merged[item_key] = compact(row, expected[item_key], source, args.method)

    missing = sorted(set(expected) - set(merged))
    if missing:
        raise RuntimeError(f"Coverage incomplete: {len(missing)} missing; first: {missing[:5]}")

    rows = [merged[item_key] for item_key in sorted(merged)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["language"], str(row["source_dataset_split"]))].append(row)
    by_group = {
        f"{language}|{split}": {
            "n": len(items),
            "correct": sum(bool(item["correct"]) for item in items),
            "accuracy": sum(bool(item["correct"]) for item in items) / len(items),
            "valid": sum(bool(item["valid"]) for item in items),
            "valid_rate": sum(bool(item["valid"]) for item in items) / len(items),
        }
        for (language, split), items in sorted(groups.items())
    }
    summary = {
        "method": args.method,
        "n": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "valid": sum(bool(row["valid"]) for row in rows),
        "valid_rate": sum(bool(row["valid"]) for row in rows) / len(rows),
        "result_source_counts": dict(Counter(row["result_source"] for row in rows)),
        "by_language_and_source_split": by_group,
        "predictions": str(prediction_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
