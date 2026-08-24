from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


INPUT_DIR = Path("artifacts/model_eval_20260821/inputs/full_low_resource_47815")
OUT_ROOT = Path("artifacts/model_eval_20260821")
MODELS = {
    "Ministral-3-8B-Instruct-2512": OUT_ROOT / "ministral3_8b_instruct_2512_bf16",
    "Llama-3.1-8B-Instruct": OUT_ROOT / "llama3_1_8b_instruct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final two-model evaluation report.")
    parser.add_argument("--output-dir", type=Path, default=OUT_ROOT / "two_model_summary")
    parser.add_argument("--skip-refresh", action="store_true", help="Do not rerun per-model summarizers before merging.")
    return parser.parse_args()


def run_per_model_summary(model_root: Path) -> Path:
    summary_dir = model_root / "summary_full_low_resource"
    cmd = [
        sys.executable,
        "scripts/summarize_model_eval_20260821.py",
        "--input-dir",
        str(INPUT_DIR),
        "--root-dir",
        str(model_root),
        "--output-dir",
        str(summary_dir),
    ]
    subprocess.run(cmd, check=True)
    return summary_dir / "metrics.json"


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"target_n": 0, "variants": []}
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def variant_rows(model_name: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant in metrics.get("variants", []):
        summary = variant.get("summary") or {}
        by_dataset = summary.get("by_dataset") or {}
        rows.append(
            {
                "model": model_name,
                "method": variant.get("label"),
                "global_mmlu": by_dataset.get("global_mmlu", {}).get("accuracy", 0.0),
                "mmlu_prox": by_dataset.get("mmlu_prox", {}).get("accuracy", 0.0),
                "overall": summary.get("accuracy", 0.0),
                "present_acc": summary.get("accuracy_on_present", 0.0),
                "coverage": summary.get("coverage", 0.0),
                "missing": summary.get("missing", 0),
                "valid_rate": summary.get("valid_rate", 0.0),
            }
        )
    return rows


def write_report(output_dir: Path, target_n: int, rows: list[dict[str, Any]], metrics_by_model: dict[str, dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Two-Model Full Low-Resource Evaluation",
        "",
        f"- target rows: {target_n}",
        "- scope: Global-MMLU and MMLU-ProX, languages bn/hi/ne/sw/te",
        "",
        "## Main Results",
        "",
        "| model | method | Global-MMLU | MMLU-ProX | Overall | present acc | coverage | missing |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {method} | {gm} | {mp} | {overall} | {present} | {coverage} | {missing} |".format(
                model=row["model"],
                method=row["method"],
                gm=pct(row["global_mmlu"]),
                mp=pct(row["mmlu_prox"]),
                overall=pct(row["overall"]),
                present=pct(row["present_acc"]),
                coverage=pct(row["coverage"]),
                missing=row["missing"],
            )
        )
    lines.extend(["", "## By Dataset And Language", ""])
    for model_name, metrics in metrics_by_model.items():
        lines.extend([f"### {model_name}", ""])
        for dataset in ["global_mmlu", "mmlu_prox"]:
            lines.extend(
                [
                    f"#### {dataset}",
                    "",
                    "| method | bn | hi | ne | sw | te |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for variant in metrics.get("variants", []):
                by_dl = (variant.get("summary") or {}).get("by_dataset_language") or {}
                cells = [pct(by_dl.get(f"{dataset}:{lang}", {}).get("accuracy", 0.0)) for lang in ["bn", "hi", "ne", "sw", "te"]]
                lines.append(f"| {variant.get('label')} | " + " | ".join(cells) + " |")
            lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    metrics_by_model: dict[str, dict[str, Any]] = {}
    for model_name, model_root in MODELS.items():
        metrics_path = model_root / "summary_full_low_resource" / "metrics.json"
        if not args.skip_refresh:
            metrics_path = run_per_model_summary(model_root)
        metrics_by_model[model_name] = load_metrics(metrics_path)
    target_n = max(int(metrics.get("target_n") or 0) for metrics in metrics_by_model.values())
    rows = []
    for model_name, metrics in metrics_by_model.items():
        rows.extend(variant_rows(model_name, metrics))
    summary = {
        "target_n": target_n,
        "models": metrics_by_model,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.output_dir, target_n, rows, metrics_by_model)
    print(json.dumps({"output_dir": str(args.output_dir), "target_n": target_n}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
