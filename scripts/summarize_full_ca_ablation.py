from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

LOW_RESOURCE_LANGS = {"bn", "hi", "ne", "sw", "te"}
DEFAULT_BASE = Path("artifacts/ca_mem/full_global_mmlu_mmlu_prox_ca_eval")
DEFAULT_ABLATION_ROOT = Path("artifacts/ca_mem/full_ca_ablation_qwen3_8b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize full low-resource CA ablations.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant as label=path1[,path2]. If omitted, known variant directories are used.",
    )
    return parser.parse_args()


def key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset") or "global_mmlu"),
        str(row.get("language") or ""),
        str(row.get("sample_id") or ""),
    )


def read_predictions(paths: list[Path]) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("language") or "") not in LOW_RESOURCE_LANGS:
                    continue
                rows[key(row)] = row
    return rows


def summarize(keys: set[tuple[str, str, str]], rows: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    present = keys & set(rows)
    n = len(keys)
    correct = sum(1 for k in present if bool(rows[k].get("correct")))
    valid = sum(1 for k in present if bool(rows[k].get("valid", True)))
    missing = n - len(present)
    summary: dict[str, Any] = {
        "target_n": n,
        "present_n": len(present),
        "missing": missing,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "accuracy_on_present": correct / len(present) if present else 0.0,
        "valid": valid,
        "valid_rate": valid / len(present) if present else 0.0,
    }
    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in sorted({k[0] for k in keys}):
        ds_keys = {k for k in keys if k[0] == dataset}
        ds_present = ds_keys & set(rows)
        ds_correct = sum(1 for k in ds_present if bool(rows[k].get("correct")))
        ds_valid = sum(1 for k in ds_present if bool(rows[k].get("valid", True)))
        by_dataset[dataset] = {
            "target_n": len(ds_keys),
            "present_n": len(ds_present),
            "missing": len(ds_keys) - len(ds_present),
            "correct": ds_correct,
            "accuracy": ds_correct / len(ds_keys) if ds_keys else 0.0,
            "accuracy_on_present": ds_correct / len(ds_present) if ds_present else 0.0,
            "valid": ds_valid,
            "valid_rate": ds_valid / len(ds_present) if ds_present else 0.0,
        }
    by_dataset_language: dict[str, dict[str, Any]] = {}
    for dataset in sorted({k[0] for k in keys}):
        for language in sorted({k[1] for k in keys if k[0] == dataset}):
            dl_keys = {k for k in keys if k[0] == dataset and k[1] == language}
            dl_present = dl_keys & set(rows)
            dl_correct = sum(1 for k in dl_present if bool(rows[k].get("correct")))
            dl_valid = sum(1 for k in dl_present if bool(rows[k].get("valid", True)))
            by_dataset_language[f"{dataset}:{language}"] = {
                "target_n": len(dl_keys),
                "present_n": len(dl_present),
                "missing": len(dl_keys) - len(dl_present),
                "correct": dl_correct,
                "accuracy": dl_correct / len(dl_keys) if dl_keys else 0.0,
                "accuracy_on_present": dl_correct / len(dl_present) if dl_present else 0.0,
                "valid": dl_valid,
                "valid_rate": dl_valid / len(dl_present) if dl_present else 0.0,
            }
    summary["by_dataset"] = by_dataset
    summary["by_dataset_language"] = by_dataset_language
    return summary


def parse_variant_arg(value: str) -> tuple[str, list[Path]]:
    label, sep, paths = value.partition("=")
    if not sep or not label.strip() or not paths.strip():
        raise ValueError(f"bad --variant value: {value!r}")
    return label.strip(), [Path(item) for item in paths.split(",") if item.strip()]


def default_variants(base_dir: Path, ablation_root: Path) -> dict[str, list[Path]]:
    variants = {
        "CA full": [
            base_dir / "ca_eval_full_bank_qwen3_8b_all_test" / "predictions.jsonl",
            base_dir / "ca_eval_full_bank_qwen3_8b_mmlu_prox_low_resource_all_test" / "predictions.jsonl",
        ]
    }
    labels = {
        "ca_no_rerank_top5": "w/o rerank",
        "ca_random_global_top5": "random cards",
        "ca_random_same_subject_top5": "same-subject random",
        "ca_top1": "top-1",
        "ca_top3": "top-3",
        "ca_top10": "top-10",
    }
    for dirname, label in labels.items():
        root = ablation_root / dirname
        variants[label] = [
            root / "predictions.jsonl",
            root / "global_plus_mmlu_prox_en" / "predictions.jsonl",
            root / "mmlu_prox_low_resource" / "predictions.jsonl",
        ]
    return variants


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = summary["variants"]
    lines = [
        "# Full CA Ablation Summary",
        "",
        f"- target rows: {summary['target_n']}",
        "- scope: Global-MMLU and MMLU-ProX, low-resource languages bn/hi/ne/sw/te",
        "",
        "| variant | Global-MMLU | MMLU-ProX | Overall | missing | valid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        by_dataset = row["summary"]["by_dataset"]
        lines.append(
            "| {label} | {gm} | {mp} | {overall} | {missing} | {valid} |".format(
                label=row["label"],
                gm=pct(by_dataset.get("global_mmlu", {}).get("accuracy", 0.0)),
                mp=pct(by_dataset.get("mmlu_prox", {}).get("accuracy", 0.0)),
                overall=pct(row["summary"].get("accuracy", 0.0)),
                missing=row["summary"].get("missing", 0),
                valid=pct(row["summary"].get("valid_rate", 0.0)),
            )
        )
    lines.extend(["", "## By Language", ""])
    for dataset in ["global_mmlu", "mmlu_prox"]:
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| variant | bn | hi | ne | sw | te |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            by_dl = row["summary"]["by_dataset_language"]
            cells = [pct(by_dl.get(f"{dataset}:{lang}", {}).get("accuracy", 0.0)) for lang in ["bn", "hi", "ne", "sw", "te"]]
            lines.append(f"| {row['label']} | " + " | ".join(cells) + " |")
        lines.append("")
    (output_dir / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    variants = default_variants(args.base_dir, args.ablation_root)
    for item in args.variant:
        label, paths = parse_variant_arg(item)
        variants[label] = paths

    loaded = {label: read_predictions(paths) for label, paths in variants.items()}
    target_keys = set(loaded["CA full"])
    if not target_keys:
        raise RuntimeError("CA full predictions are missing; cannot define target key set")

    rows = []
    for label, paths in variants.items():
        rows.append(
            {
                "label": label,
                "paths": [str(path) for path in paths],
                "summary": summarize(target_keys, loaded[label]),
            }
        )
    summary = {
        "target_n": len(target_keys),
        "low_resource_languages": sorted(LOW_RESOURCE_LANGS),
        "variants": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir, summary)
    print(json.dumps({"output_dir": str(args.output_dir), "target_n": len(target_keys)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
