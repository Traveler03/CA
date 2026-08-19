from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_key(row: dict[str, Any]) -> str:
    dataset = str(row.get("_dataset") or row.get("dataset") or "global_mmlu")
    language = str(row.get("language") or row.get("_language") or "")
    sample_id = str(row.get("sample_id") or row.get("question_id") or row.get("id") or "")
    return f"{dataset}|{language}|{sample_id}"


def card_hit_matches_sample(row: dict[str, Any]) -> bool:
    sample_id = str(row.get("sample_id") or "")
    for card in row.get("retrieved_cards") or []:
        source_ids = card.get("source_sample_ids")
        if isinstance(source_ids, list) and sample_id in {str(item) for item in source_ids}:
            return True
    return False


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    exp = args.experiment_dir
    ref_path = exp / "zero_wrong100_zero_shot_reference.jsonl"
    eval_path = exp / "zero_wrong100_eval_input.jsonl"
    refs = read_jsonl(ref_path)
    eval_rows = read_jsonl(eval_path)
    ca_rows = read_jsonl(args.ca_predictions)

    eval_by_key = {row_key(row): row for row in eval_rows}
    ca_by_key = {row_key(row): row for row in ca_rows}
    ref_by_key = {str(row.get("eval_key_ca")): row for row in refs}
    common_keys = sorted(set(eval_by_key) & set(ca_by_key))

    rows = []
    for key in common_keys:
        eval_row = eval_by_key[key]
        ca = ca_by_key[key]
        ref = ref_by_key.get(key, {})
        rows.append(
            {
                "key": key,
                "dataset": eval_row.get("_dataset") or eval_row.get("dataset") or "global_mmlu",
                "language": eval_row.get("language") or eval_row.get("_language"),
                "sample_id": eval_row.get("sample_id"),
                "subject": eval_row.get("subject"),
                "gold": eval_row.get("answer"),
                "zero_prediction": ref.get("prediction") or eval_row.get("_zero_shot_prediction"),
                "ca_prediction": ca.get("prediction"),
                "ca_correct": bool(ca.get("correct")),
                "ca_valid": bool(ca.get("valid")),
                "exact_source_card_retrieved": card_hit_matches_sample(ca),
                "retrieved_card_count": len(ca.get("retrieved_cards") or []),
            }
        )

    total = len(rows)
    ca_correct = sum(1 for row in rows if row["ca_correct"])
    valid = sum(1 for row in rows if row["ca_valid"])
    exact_hits = sum(1 for row in rows if row["exact_source_card_retrieved"])
    by_language = []
    for language in sorted({str(row["language"]) for row in rows}):
        items = [row for row in rows if str(row["language"]) == language]
        by_language.append(
            {
                "language": language,
                "n": len(items),
                "ca_correct": sum(1 for row in items if row["ca_correct"]),
                "ca_accuracy": sum(1 for row in items if row["ca_correct"]) / len(items) if items else 0.0,
                "exact_source_card_retrieved": sum(1 for row in items if row["exact_source_card_retrieved"]),
            }
        )
    by_subject = []
    for subject, count in Counter(str(row["subject"]) for row in rows).most_common():
        items = [row for row in rows if str(row["subject"]) == subject]
        by_subject.append(
            {
                "subject": subject,
                "n": count,
                "ca_correct": sum(1 for row in items if row["ca_correct"]),
                "ca_accuracy": sum(1 for row in items if row["ca_correct"]) / len(items) if items else 0.0,
            }
        )
    summary = {
        "n_reference": len(refs),
        "n_eval_input": len(eval_rows),
        "n_ca_predictions": len(ca_rows),
        "n_common": total,
        "zero_shot_selected_accuracy": 0.0,
        "ca_correct": ca_correct,
        "ca_accuracy_on_zero_wrong100": ca_correct / total if total else 0.0,
        "ca_valid": valid,
        "ca_parse_rate": valid / total if total else 0.0,
        "exact_source_card_retrieved": exact_hits,
        "exact_source_card_retrieval_rate": exact_hits / total if total else 0.0,
        "by_language": by_language,
        "by_subject": by_subject,
        "ca_predictions": str(args.ca_predictions),
    }
    out_json = exp / "zero_wrong100_card_ca_comparison.json"
    out_md = exp / "zero_wrong100_card_ca_comparison.md"
    out_rows = exp / "zero_wrong100_card_ca_items.jsonl"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with out_rows.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    out_md.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Zero-shot Wrong100 Oracle-card CA Comparison",
        "",
        f"- common rows: {summary['n_common']}",
        f"- zero-shot selected accuracy: {summary['zero_shot_selected_accuracy']:.2%}",
        f"- CA accuracy on selected zero-shot-wrong rows: {summary['ca_accuracy_on_zero_wrong100']:.2%} ({summary['ca_correct']}/{summary['n_common']})",
        f"- CA parse rate: {summary['ca_parse_rate']:.2%} ({summary['ca_valid']}/{summary['n_common']})",
        f"- exact source card retrieved: {summary['exact_source_card_retrieval_rate']:.2%} ({summary['exact_source_card_retrieved']}/{summary['n_common']})",
        "",
        "## By language",
        "",
        "| language | n | CA correct | CA acc | exact card hit |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["by_language"]:
        lines.append(
            f"| {row['language']} | {row['n']} | {row['ca_correct']} | {row['ca_accuracy']:.2%} | {row['exact_source_card_retrieved']} |"
        )
    lines.extend(["", "## By subject", "", "| subject | n | CA correct | CA acc |", "|---|---:|---:|---:|"])
    for row in summary["by_subject"][:30]:
        lines.append(f"| {row['subject']} | {row['n']} | {row['ca_correct']} | {row['ca_accuracy']:.2%} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare CA predictions against a selected zero-shot-wrong100 baseline.")
    parser.add_argument("--experiment-dir", type=Path, default=Path("artifacts/ca_mem/zero_wrong100_global_mmlu_oracle_card_ca_gpt54_v1"))
    parser.add_argument("--ca-predictions", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
