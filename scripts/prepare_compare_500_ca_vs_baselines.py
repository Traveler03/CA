from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_ROOT = Path("artifacts/ca_mem/compare_500_ca_vs_baselines")
DEFAULT_GLOBAL_DATA = Path("data/processed/global_mmlu/en.jsonl")
DEFAULT_MMLU_PROX_DATA = Path("data/processed/mmlu_prox/en.jsonl")
DEFAULT_GLOBAL_BANK = Path(
    "artifacts/ca_mem/global_mmlu_oracle_cards_en_gpt54_v1/banks/concept_card_qwen3_1024"
)
DEFAULT_MMLU_PROX_BANK = Path(
    "artifacts/ca_mem/mmlu_prox_non_global_oracle_cards_en_gpt54_v1/banks/concept_card_qwen3_1024"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def normalized_eval_row(row: dict[str, Any], *, dataset: str) -> dict[str, Any]:
    out = dict(row)
    out["_dataset"] = dataset
    out["language"] = "en"
    if "answer" not in out and out.get("answer_label"):
        out["answer"] = str(out["answer_label"]).strip().upper()
    return out


def sample_rows(
    *,
    global_data: Path,
    mmlu_prox_data: Path,
    global_n: int,
    mmlu_prox_n: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    global_rows = [normalized_eval_row(row, dataset="global_mmlu") for row in read_jsonl(global_data)]
    mmlu_rows = [
        normalized_eval_row(row, dataset="mmlu_prox")
        for row in read_jsonl(mmlu_prox_data)
        if row.get("excluded_global_mmlu_overlap") is False
    ]
    if len(global_rows) < global_n:
        raise ValueError(f"Global-MMLU has only {len(global_rows)} rows, need {global_n}")
    if len(mmlu_rows) < mmlu_prox_n:
        raise ValueError(f"MMLU-ProX non-overlap has only {len(mmlu_rows)} rows, need {mmlu_prox_n}")
    selected_global = rng.sample(global_rows, global_n)
    selected_mmlu = rng.sample(mmlu_rows, mmlu_prox_n)
    mixed = selected_global + selected_mmlu
    rng.shuffle(mixed)
    manifest = {
        "seed": seed,
        "global_mmlu_source": str(global_data),
        "mmlu_prox_source": str(mmlu_prox_data),
        "global_mmlu_available": len(global_rows),
        "mmlu_prox_non_overlap_available": len(mmlu_rows),
        "global_mmlu_sampled": len(selected_global),
        "mmlu_prox_sampled": len(selected_mmlu),
        "total_sampled": len(mixed),
        "datasets": {
            "global_mmlu": sorted(str(row.get("sample_id")) for row in selected_global),
            "mmlu_prox": sorted(str(row.get("sample_id")) for row in selected_mmlu),
        },
    }
    return mixed, manifest


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def merge_runtime_banks(
    *,
    global_bank: Path,
    mmlu_prox_bank: Path,
    output_bank: Path,
) -> dict[str, Any]:
    output_bank.mkdir(parents=True, exist_ok=True)
    sources = [
        ("global_mmlu_oracle", global_bank),
        ("mmlu_prox_oracle", mmlu_prox_bank),
    ]
    bank_out = output_bank / "bank.jsonl"
    arrays = []
    source_counts: dict[str, int] = {}
    with bank_out.open("w", encoding="utf-8") as out:
        for name, bank_dir in sources:
            bank_path = bank_dir / "bank.jsonl"
            index_path = bank_dir / "build_index.npy"
            if not bank_path.exists():
                raise FileNotFoundError(bank_path)
            if not index_path.exists():
                raise FileNotFoundError(index_path)
            count = count_jsonl(bank_path)
            arr = np.load(index_path, mmap_mode="r")
            if int(arr.shape[0]) != count:
                raise ValueError(f"{name} index/card mismatch: {arr.shape[0]} != {count}")
            source_counts[name] = count
            arrays.append(np.asarray(arr, dtype=np.float32))
            with bank_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
    merged = np.concatenate(arrays, axis=0)
    np.save(output_bank / "build_index.npy", merged)
    manifest = {
        "bank_dir": str(output_bank),
        "bank_jsonl": str(bank_out),
        "embedding_index": str(output_bank / "build_index.npy"),
        "card_count": int(merged.shape[0]),
        "embedding_shape": [int(v) for v in merged.shape],
        "source_counts": source_counts,
        "contains_oracle_cards": True,
        "diagnostic_only": True,
        "can_be_used_for_clean_eval": False,
    }
    (output_bank / "runtime_bank_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the 500-row CA-vs-baseline comparison set.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--global-data", type=Path, default=DEFAULT_GLOBAL_DATA)
    parser.add_argument("--mmlu-prox-data", type=Path, default=DEFAULT_MMLU_PROX_DATA)
    parser.add_argument("--global-bank", type=Path, default=DEFAULT_GLOBAL_BANK)
    parser.add_argument("--mmlu-prox-bank", type=Path, default=DEFAULT_MMLU_PROX_BANK)
    parser.add_argument("--global-n", type=int, default=250)
    parser.add_argument("--mmlu-prox-n", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows, sample_manifest = sample_rows(
        global_data=args.global_data,
        mmlu_prox_data=args.mmlu_prox_data,
        global_n=args.global_n,
        mmlu_prox_n=args.mmlu_prox_n,
        seed=args.seed,
    )
    input_path = args.output_root / "input_500_mixed.jsonl"
    write_jsonl(input_path, rows)
    bank_manifest = merge_runtime_banks(
        global_bank=args.global_bank,
        mmlu_prox_bank=args.mmlu_prox_bank,
        output_bank=args.output_root / "merged_oracle_runtime_bank_qwen3_1024",
    )
    manifest = {
        **sample_manifest,
        "input_jsonl": str(input_path),
        "merged_bank": bank_manifest,
        "oracle_card_warning": (
            "CA-card uses cards generated from each benchmark question and gold answer; "
            "use these results only as diagnostic upper-bound behavior, not clean evaluation."
        ),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
