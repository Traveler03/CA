from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("artifacts/ca_mem/lowres500_global_mmlu_ca_check")
DEFAULT_GLOBAL_DIR = Path("data/processed/global_mmlu")
DEFAULT_LANGUAGES = ["bn", "hi", "ne", "sw", "te"]
DEFAULT_ZERO = Path("runs/smoke_001/full_qwen3_8b_localized_zero_shot_2gpu/predictions.jsonl")
DEFAULT_TCRAG = Path("runs/smoke_001/full_tcrag_qwen3_8b_local_2gpu/predictions.jsonl")
DEFAULT_CORAL = Path("runs/smoke_001/full_coral_wikipag_raw_nofallback_qwen3_8b_local_2gpu/predictions.jsonl")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def eval_key(row: dict[str, Any]) -> str:
    return f"global_mmlu|{row.get('language')}|{row.get('sample_id')}"


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    english_rows = {str(row["sample_id"]): row for row in read_jsonl(args.global_mmlu_dir / "en.test.jsonl")}
    localized_by_language = {
        language: {str(row["sample_id"]): row for row in read_jsonl(args.global_mmlu_dir / f"{language}.test.jsonl")}
        for language in args.languages
    }
    common_ids = sorted(
        set(english_rows).intersection(*(set(rows) for rows in localized_by_language.values()))
    )
    if len(common_ids) < args.source_n:
        raise RuntimeError(f"only {len(common_ids)} common test IDs available, need {args.source_n}")
    selected_ids = sorted(rng.sample(common_ids, args.source_n))

    eval_rows: list[dict[str, Any]] = []
    for sample_id in selected_ids:
        for language in args.languages:
            row = dict(localized_by_language[language][sample_id])
            row["_dataset"] = "global_mmlu"
            row["dataset"] = "global_mmlu"
            row["language"] = language
            eval_rows.append(row)

    card_rows: list[dict[str, Any]] = []
    for sample_id in selected_ids:
        row = dict(english_rows[sample_id])
        row["_dataset"] = "global_mmlu"
        row["dataset"] = "global_mmlu"
        row["language"] = "en"
        card_rows.append(row)

    write_jsonl(out / "lowres500_eval_input.jsonl", eval_rows)
    write_jsonl(out / "lowres500_card_input_en.jsonl", card_rows)

    manifest = {
        "dataset": "global_mmlu",
        "seed": args.seed,
        "source_n": len(selected_ids),
        "row_n": len(eval_rows),
        "languages": args.languages,
        "rows_per_language": dict(sorted(Counter(row["language"] for row in eval_rows).items())),
        "global_mmlu_dir": str(args.global_mmlu_dir),
        "eval_input_jsonl": str(out / "lowres500_eval_input.jsonl"),
        "card_input_jsonl": str(out / "lowres500_card_input_en.jsonl"),
        "selected_sample_ids": selected_ids,
        "note": "100 Global-MMLU test source IDs expanded to five low-resource languages; cards use English question+gold answer.",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


_FIELD_PATTERNS = {
    "dataset": re.compile(r'"dataset"\s*:\s*"([^"]*)"'),
    "language": re.compile(r'"language"\s*:\s*"([^"]*)"'),
    "sample_id": re.compile(r'"sample_id"\s*:\s*"([^"]*)"'),
    "prediction": re.compile(r'"prediction"\s*:\s*"([^"]*)"'),
    "answer": re.compile(r'"answer"\s*:\s*"([^"]*)"'),
    "correct": re.compile(r'"correct"\s*:\s*(true|false)'),
    "valid": re.compile(r'"valid"\s*:\s*(true|false)'),
}


def field(line: str, name: str) -> str | None:
    match = _FIELD_PATTERNS[name].search(line)
    return match.group(1) if match else None


def load_predictions_fast(path: Path, wanted: set[tuple[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            if field(line, "dataset") != "global_mmlu":
                continue
            language = field(line, "language")
            sample_id = field(line, "sample_id")
            if language is None or sample_id is None:
                continue
            key = (language, sample_id)
            if key not in wanted:
                continue
            rows[key] = {
                "dataset": "global_mmlu",
                "language": language,
                "sample_id": sample_id,
                "prediction": field(line, "prediction"),
                "answer": field(line, "answer"),
                "correct": field(line, "correct") == "true",
                "valid": field(line, "valid") != "false",
            }
    return rows


def load_ca_predictions(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = {}
    for row in read_jsonl(path):
        if str(row.get("dataset")) != "global_mmlu":
            continue
        rows[(str(row.get("language")), str(row.get("sample_id")))] = row
    return rows


def summarize_method(name: str, preds: dict[tuple[str, str], dict[str, Any]], wanted: set[tuple[str, str]]) -> dict[str, Any]:
    present = [preds[key] for key in sorted(wanted) if key in preds]
    by_language = {}
    for language in sorted({key[0] for key in wanted}):
        items = [preds[key] for key in sorted(wanted) if key in preds and key[0] == language]
        by_language[language] = {
            "n": len(items),
            "correct": sum(1 for row in items if row.get("correct")),
            "accuracy": sum(1 for row in items if row.get("correct")) / len(items) if items else 0.0,
        }
    correct = sum(1 for row in present if row.get("correct"))
    return {
        "method": name,
        "n": len(present),
        "missing": len(wanted) - len(present),
        "correct": correct,
        "accuracy": correct / len(present) if present else 0.0,
        "by_language": by_language,
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    eval_rows = read_jsonl(args.output_dir / "lowres500_eval_input.jsonl")
    wanted = {(str(row["language"]), str(row["sample_id"])) for row in eval_rows}
    methods = {
        "zero_shot": load_predictions_fast(args.zero_shot_predictions, wanted),
        "tCRAG": load_predictions_fast(args.tcrag_predictions, wanted),
        "CORAL-Wikipag": load_predictions_fast(args.coral_predictions, wanted),
        "CA-oracle": load_ca_predictions(args.ca_predictions),
    }
    summaries = [summarize_method(name, preds, wanted) for name, preds in methods.items()]
    by_key: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for name, preds in methods.items():
        for key, row in preds.items():
            if key in wanted:
                by_key[key][name] = row

    pairwise = {}
    ca_preds = methods["CA-oracle"]
    for baseline_name in ["zero_shot", "tCRAG", "CORAL-Wikipag"]:
        baseline = methods[baseline_name]
        common = sorted(set(ca_preds) & set(baseline) & wanted)
        ca_correct = sum(1 for key in common if ca_preds[key].get("correct"))
        base_correct = sum(1 for key in common if baseline[key].get("correct"))
        ca_win = sum(1 for key in common if ca_preds[key].get("correct") and not baseline[key].get("correct"))
        ca_loss = sum(1 for key in common if baseline[key].get("correct") and not ca_preds[key].get("correct"))
        pairwise[baseline_name] = {
            "common_n": len(common),
            "ca_correct": ca_correct,
            "baseline_correct": base_correct,
            "ca_accuracy": ca_correct / len(common) if common else 0.0,
            "baseline_accuracy": base_correct / len(common) if common else 0.0,
            "delta_accuracy": (ca_correct - base_correct) / len(common) if common else 0.0,
            "ca_win_baseline_wrong": ca_win,
            "ca_loss_baseline_correct": ca_loss,
        }

    result = {
        "eval_rows": len(eval_rows),
        "wanted_keys": len(wanted),
        "summaries": summaries,
        "pairwise_vs_ca": pairwise,
        "paths": {
            "ca_predictions": str(args.ca_predictions),
            "zero_shot_predictions": str(args.zero_shot_predictions),
            "tcrag_predictions": str(args.tcrag_predictions),
            "coral_predictions": str(args.coral_predictions),
        },
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Low-resource 500 Global-MMLU CA comparison",
        "",
        f"- rows: {len(eval_rows)}",
        f"- unique source IDs: {len({row['sample_id'] for row in eval_rows})}",
        f"- languages: {', '.join(sorted({row['language'] for row in eval_rows}))}",
        "",
        "| method | n | correct | accuracy | missing |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(f"| {item['method']} | {item['n']} | {item['correct']} | {item['accuracy']:.2%} | {item['missing']} |")
    lines.extend(["", "## Pairwise vs CA", "", "| baseline | common n | CA | baseline | delta | CA wins | CA losses |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, item in pairwise.items():
        lines.append(
            f"| {name} | {item['common_n']} | {item['ca_accuracy']:.2%} | "
            f"{item['baseline_accuracy']:.2%} | {item['delta_accuracy']:.2%} | "
            f"{item['ca_win_baseline_wrong']} | {item['ca_loss_baseline_correct']} |"
        )
    (args.output_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and compare a low-resource Global-MMLU 500-row CA eval.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--global-mmlu-dir", type=Path, default=DEFAULT_GLOBAL_DIR)
    parser.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    parser.add_argument("--source-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero-shot-predictions", type=Path, default=DEFAULT_ZERO)
    parser.add_argument("--tcrag-predictions", type=Path, default=DEFAULT_TCRAG)
    parser.add_argument("--coral-predictions", type=Path, default=DEFAULT_CORAL)
    parser.add_argument("--ca-predictions", type=Path)
    parser.add_argument("--compare", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.compare:
        if args.ca_predictions is None:
            raise SystemExit("--ca-predictions is required with --compare")
        print(json.dumps(compare(args), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(prepare(args), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
