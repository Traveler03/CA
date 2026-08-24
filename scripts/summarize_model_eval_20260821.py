from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("artifacts/model_eval_20260821/ministral3_8b_instruct_2512_bf16")
DEFAULT_INPUT_DIR = Path("artifacts/model_eval_20260821/inputs/full_low_resource_47815")
LOW_RESOURCE_LANGS = ["bn", "hi", "ne", "sw", "te"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize full low-resource model evaluation outputs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "summary_full_low_resource")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Extra variant as label=path1[,path2]. Rows are keyed by dataset/language/sample_id.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                yield {"_json_error": f"{path}:{line_no}: {exc}"}


def split_eval_key(value: Any) -> tuple[str | None, str | None, str | None, str | None]:
    parts = str(value or "").split("|")
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2], "|".join(parts[3:])
    if len(parts) == 3:
        return None, parts[0], parts[1], parts[2]
    return None, None, None, None


def target_key(row: dict[str, Any]) -> tuple[str, str, str] | None:
    _, key_dataset, key_language, key_sample = split_eval_key(row.get("eval_key"))
    dataset = str(row.get("_dataset") or row.get("dataset") or key_dataset or "").strip()
    language = str(row.get("language") or row.get("_language") or key_language or "").strip()
    sample_id = str(row.get("sample_id") or row.get("question_id") or row.get("id") or key_sample or "").strip()
    if not dataset or not language or not sample_id:
        return None
    return dataset, language, sample_id


def read_targets(input_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    targets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(input_dir.glob("*.jsonl")):
        for row in iter_jsonl(path):
            key = target_key(row)
            if key is None:
                continue
            targets[key] = row
    return targets


def row_correct(row: dict[str, Any], target: dict[str, Any] | None = None) -> bool:
    value = row.get("correct")
    if value is not None:
        return bool(value)
    pred = str(row.get("prediction") or row.get("pred_answer") or row.get("answer_payload", {}).get("answer") or "").strip().upper()
    gold = str(row.get("answer") or row.get("gold") or (target or {}).get("answer") or "").strip().upper()
    return bool(pred and gold and pred[:1] == gold[:1])


def parse_variant_arg(value: str) -> tuple[str, list[Path]]:
    label, sep, paths = value.partition("=")
    if not sep or not label.strip() or not paths.strip():
        raise ValueError(f"bad --variant value: {value!r}")
    return label.strip(), [Path(item) for item in paths.split(",") if item.strip()]


def default_variants(root_dir: Path) -> dict[str, list[Path]]:
    return {
        "zero_shot": [
            root_dir / "baselines_shard0" / "predictions.jsonl",
            root_dir / "baselines_shard1" / "predictions.jsonl",
        ],
        "tCRAG": [
            root_dir / "baselines_shard0" / "predictions.jsonl",
            root_dir / "baselines_shard1" / "predictions.jsonl",
        ],
        "CORAL-Wikipag": [
            root_dir / "baselines_shard0" / "predictions.jsonl",
            root_dir / "baselines_shard1" / "predictions.jsonl",
        ],
        "CA": [
            root_dir / "ca_shard0" / "predictions.jsonl",
            root_dir / "ca_shard1" / "predictions.jsonl",
        ],
    }


def label_matches(label: str, row: dict[str, Any]) -> bool:
    method = str(row.get("method") or split_eval_key(row.get("eval_key"))[0] or "")
    if label == "zero_shot":
        return method == "zero_shot"
    if label == "tCRAG":
        return method in {"trag", "tcrag", "tCRAG"}
    if label == "CORAL-Wikipag":
        return method == "coral_wikipag"
    if label == "CA":
        return method.startswith("ca_concept_bank")
    return True


def read_predictions(label: str, paths: list[Path]) -> dict[str, Any]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    stats: dict[str, Any] = {
        "paths": [str(path) for path in paths],
        "paths_existing": [str(path) for path in paths if path.exists()],
        "rows_read": 0,
        "error_rows": 0,
        "json_error_rows": 0,
        "matched_rows": 0,
        "duplicate_valid_rows": 0,
    }
    for path in paths:
        for row in iter_jsonl(path):
            stats["rows_read"] += 1
            if not label_matches(label, row):
                continue
            if row.get("_json_error"):
                stats["json_error_rows"] += 1
                continue
            if row.get("error"):
                stats["error_rows"] += 1
                continue
            key = target_key(row)
            if key is None:
                continue
            if key in rows:
                stats["duplicate_valid_rows"] += 1
            rows[key] = row
            stats["matched_rows"] += 1
    return {"rows": rows, "stats": stats}


def empty_bucket() -> dict[str, Any]:
    return {"target_n": 0, "present_n": 0, "missing": 0, "correct": 0, "valid": 0}


def finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    target_n = int(bucket["target_n"])
    present_n = int(bucket["present_n"])
    correct = int(bucket["correct"])
    valid = int(bucket["valid"])
    bucket["missing"] = target_n - present_n
    bucket["accuracy"] = correct / target_n if target_n else 0.0
    bucket["accuracy_on_present"] = correct / present_n if present_n else 0.0
    bucket["coverage"] = present_n / target_n if target_n else 0.0
    bucket["valid_rate"] = valid / present_n if present_n else 0.0
    return bucket


def summarize_variant(
    targets: dict[tuple[str, str, str], dict[str, Any]],
    predictions: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    overall = empty_bucket()
    by_dataset: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    by_language: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    by_dataset_language: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    by_subject: dict[str, dict[str, Any]] = defaultdict(empty_bucket)

    for key, target in targets.items():
        dataset, language, _sample_id = key
        subject = str(target.get("subject") or "")
        groups = [
            overall,
            by_dataset[dataset],
            by_language[language],
            by_dataset_language[f"{dataset}:{language}"],
            by_subject[subject],
        ]
        for bucket in groups:
            bucket["target_n"] += 1
        row = predictions.get(key)
        if row is None:
            continue
        correct = row_correct(row, target)
        valid = bool(row.get("valid", True))
        for bucket in groups:
            bucket["present_n"] += 1
            bucket["correct"] += int(correct)
            bucket["valid"] += int(valid)

    return {
        **finalize_bucket(overall),
        "by_dataset": {name: finalize_bucket(bucket) for name, bucket in sorted(by_dataset.items())},
        "by_language": {name: finalize_bucket(bucket) for name, bucket in sorted(by_language.items())},
        "by_dataset_language": {name: finalize_bucket(bucket) for name, bucket in sorted(by_dataset_language.items())},
        "by_subject": {name: finalize_bucket(bucket) for name, bucket in sorted(by_subject.items())},
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Full Low-Resource Evaluation",
        "",
        f"- target rows: {summary['target_n']}",
        "- scope: Global-MMLU and MMLU-ProX, languages bn/hi/ne/sw/te",
        "",
        "## Main",
        "",
        "| method | Global-MMLU | MMLU-ProX | Overall | present acc | coverage | missing | valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["variants"]:
        metrics = row["summary"]
        by_dataset = metrics["by_dataset"]
        lines.append(
            "| {label} | {gm} | {mp} | {overall} | {present} | {coverage} | {missing} | {valid} |".format(
                label=row["label"],
                gm=pct(by_dataset.get("global_mmlu", {}).get("accuracy", 0.0)),
                mp=pct(by_dataset.get("mmlu_prox", {}).get("accuracy", 0.0)),
                overall=pct(metrics.get("accuracy", 0.0)),
                present=pct(metrics.get("accuracy_on_present", 0.0)),
                coverage=pct(metrics.get("coverage", 0.0)),
                missing=metrics.get("missing", 0),
                valid=pct(metrics.get("valid_rate", 0.0)),
            )
        )
    lines.extend(["", "## Dataset By Language", ""])
    for dataset in ["global_mmlu", "mmlu_prox"]:
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| method | bn | hi | ne | sw | te |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["variants"]:
            by_dl = row["summary"]["by_dataset_language"]
            cells = [pct(by_dl.get(f"{dataset}:{lang}", {}).get("accuracy", 0.0)) for lang in LOW_RESOURCE_LANGS]
            lines.append(f"| {row['label']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(["## Coverage", "", "| method | matched rows | duplicate valid rows | error rows | files |", "|---|---:|---:|---:|---|"])
    for row in summary["variants"]:
        stats = row["stats"]
        files = "<br>".join(stats.get("paths_existing") or [])
        lines.append(
            f"| {row['label']} | {stats.get('matched_rows', 0)} | {stats.get('duplicate_valid_rows', 0)} | "
            f"{stats.get('error_rows', 0)} | {files} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    variants = default_variants(args.root_dir)
    for item in args.variant:
        label, paths = parse_variant_arg(item)
        variants[label] = paths

    targets = read_targets(args.input_dir)
    rows = []
    for label, paths in variants.items():
        loaded = read_predictions(label, paths)
        rows.append(
            {
                "label": label,
                "paths": [str(path) for path in paths],
                "stats": loaded["stats"],
                "summary": summarize_variant(targets, loaded["rows"]),
            }
        )
    summary = {
        "target_n": len(targets),
        "low_resource_languages": LOW_RESOURCE_LANGS,
        "variants": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir, summary)
    print(json.dumps({"output_dir": str(args.output_dir), "target_n": len(targets)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
