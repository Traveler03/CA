from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.jsonl import read_jsonl, write_jsonl
from scripts.run_multilingual_rag_baselines import summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded multilingual RAG prediction directories.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--method-name", default="")
    return parser.parse_args()


def load_shard_predictions(shard_dirs: list[Path]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    source_counts: dict[str, dict[str, int]] = {}
    for shard_dir in shard_dirs:
        prediction_path = shard_dir / "predictions.jsonl"
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        rows = list(read_jsonl(prediction_path))
        duplicates = 0
        for row in rows:
            key = str(row.get("eval_key") or "")
            if not key:
                key = f"{row.get('method')}|{row.get('dataset')}|{row.get('language')}|{row.get('sample_id')}|{len(rows_by_key)}"
            if key in rows_by_key:
                duplicates += 1
                continue
            rows_by_key[key] = row
        source_counts[str(prediction_path)] = {
            "rows": len(rows),
            "duplicates_skipped": duplicates,
            "errors": sum(1 for row in rows if row.get("error")),
        }
    return list(rows_by_key.values()), source_counts


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, source_counts = load_shard_predictions(args.shard_dirs)
    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    summary = summarize(rows)
    summary["merge"] = {
        "method_name": args.method_name,
        "shard_dirs": [str(path) for path in args.shard_dirs],
        "source_counts": source_counts,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# Merged multilingual RAG result",
        "",
        f"- rows: {summary['n']}",
        f"- correct: {summary['correct']}",
        f"- accuracy: {summary['accuracy']:.4f}",
        f"- parse_rate: {summary['parse_rate']:.4f}",
        "",
        "| method | n | accuracy | parse_rate |",
        "|---|---:|---:|---:|",
    ]
    for item in summary["by_method"]:
        report_lines.append(f"| {item['method']} | {item['n']} | {item['accuracy']:.4f} | {item['parse_rate']:.4f} |")
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"event": "merged", "summary": summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
