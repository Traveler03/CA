from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_usage_bank import audit as audit_bank
from scripts.combine_clean_shards import combine as combine_clean
from scripts.merge_usage_banks import merge as merge_banks


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_complete_bank(path: Path) -> bool:
    return (path / "summary.json").exists() and (path / "bank_manifest.json").exists()


def readiness(oracle_dir: Path | None, mmlu_prox_oracle_dir: Path | None, shard_dirs: list[Path]) -> dict[str, Any]:
    return {
        "oracle_complete": is_complete_bank(oracle_dir) if oracle_dir is not None else None,
        "mmlu_prox_oracle_complete": is_complete_bank(mmlu_prox_oracle_dir) if mmlu_prox_oracle_dir is not None else None,
        "shards": [
            {
                "path": str(shard),
                "complete": is_complete_bank(shard),
                "completed_subjects": len(list((shard / "subjects").glob("*/summary.json"))) if (shard / "subjects").exists() else 0,
            }
            for shard in shard_dirs
        ],
    }


def final_audit_ok(audit_path: Path) -> bool:
    if not audit_path.exists():
        return False
    try:
        return bool(load_json(audit_path).get("ok"))
    except Exception:
        return False


def finalize_once(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    oracle_dir = Path(args.oracle_dir) if args.oracle_dir else None
    mmlu_prox_oracle_dir = Path(args.mmlu_prox_oracle_dir) if args.mmlu_prox_oracle_dir else None
    clean_combined_dir = Path(args.clean_combined_dir)
    combined_dir = Path(args.combined_dir) if args.combined_dir else None
    shard_dirs = [Path(item) for item in args.shard_dirs]

    state = readiness(oracle_dir, mmlu_prox_oracle_dir, shard_dirs)
    if not args.clean_only and not state["oracle_complete"]:
        return {"ok": False, "stage": "waiting", "root": str(root), "readiness": state}
    if not args.clean_only and mmlu_prox_oracle_dir is not None and not state["mmlu_prox_oracle_complete"]:
        return {"ok": False, "stage": "waiting", "root": str(root), "readiness": state}
    if not all(item["complete"] for item in state["shards"]):
        return {"ok": False, "stage": "waiting", "root": str(root), "readiness": state}

    if not is_complete_bank(clean_combined_dir):
        clean_manifest = combine_clean(
            SimpleNamespace(
                shard_dirs=[str(path) for path in shard_dirs],
                output_dir=str(clean_combined_dir),
                bank_version=args.clean_bank_version,
                write_runtime_bank=True,
                build_runtime_embedding_index=True,
                embedding_model_dir=args.embedding_model_dir,
                embedding_device=args.embedding_device,
                embedding_truncate_dim=args.embedding_truncate_dim,
                embedding_batch_size=args.embedding_batch_size,
            )
        )
    else:
        clean_manifest = load_json(clean_combined_dir / "bank_manifest.json")

    if args.clean_only:
        audit_path = clean_combined_dir / "audit.json"
        audit_result = audit_bank(SimpleNamespace(bank_dir=str(clean_combined_dir), output=str(audit_path)))
        return {
            "ok": bool(audit_result.get("ok")),
            "stage": "finalized",
            "root": str(root),
            "clean_combined_dir": str(clean_combined_dir),
            "clean_subjects": clean_manifest.get("subjects"),
            "clean_cards": clean_manifest.get("active_card_count"),
            "runtime_bank": clean_manifest.get("runtime_bank"),
            "audit_path": str(audit_path),
        }

    if combined_dir is None or oracle_dir is None:
        raise ValueError("combined_dir and oracle_dir are required unless --clean-only is set")
    if not is_complete_bank(combined_dir):
        components = [
            f"global_mmlu_oracle={oracle_dir}",
            f"wikipag_clean={clean_combined_dir}",
        ]
        if mmlu_prox_oracle_dir is not None:
            components.insert(1, f"mmlu_prox_oracle={mmlu_prox_oracle_dir}")
        combined_manifest = merge_banks(
            SimpleNamespace(
                oracle_dir=None,
                clean_dir=None,
                output_dir=str(combined_dir),
                bank_version=args.combined_bank_version,
                component=components,
                oracle_component="global_mmlu_oracle",
                clean_component="wikipag_clean",
                write_runtime_bank=True,
                build_runtime_embedding_index=True,
                embedding_model_dir=args.embedding_model_dir,
                embedding_device=args.embedding_device,
                embedding_truncate_dim=args.embedding_truncate_dim,
                embedding_batch_size=args.embedding_batch_size,
            )
        )
    else:
        combined_manifest = load_json(combined_dir / "bank_manifest.json")

    audit_path = combined_dir / "audit.json"
    audit_result = audit_bank(SimpleNamespace(bank_dir=str(combined_dir), output=str(audit_path)))
    return {
        "ok": bool(audit_result.get("ok")),
        "stage": "finalized",
        "root": str(root),
        "oracle_dir": str(oracle_dir),
        "mmlu_prox_oracle_dir": str(mmlu_prox_oracle_dir) if mmlu_prox_oracle_dir is not None else None,
        "clean_combined_dir": str(clean_combined_dir),
        "combined_dir": str(combined_dir),
        "clean_subjects": clean_manifest.get("subjects"),
        "clean_cards": clean_manifest.get("active_card_count"),
        "combined_subjects": combined_manifest.get("subjects"),
        "combined_cards": combined_manifest.get("usage_card_count"),
        "runtime_bank": combined_manifest.get("runtime_bank"),
        "audit_path": str(audit_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    last_state: dict[str, Any] | None = None
    while True:
        state = finalize_once(args)
        last_state = state
        print(json.dumps(state, ensure_ascii=False, sort_keys=True), flush=True)
        if state.get("ok") or not args.wait:
            return state
        if state.get("stage") == "finalized" and not state.get("ok"):
            return state
        time.sleep(args.poll_interval_s)


def default_shard_dirs(root: Path) -> list[str]:
    return [
        str(root / "wikipag_clean_all_subjects_target1000_gpt54"),
        *[str(root / f"wikipag_clean_shard{i}_target1000_gpt54") for i in range(1, 6)],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finalize the full oracle + clean Usage Bank run once all components finish.")
    parser.add_argument("--root", default="runs/full_20260730")
    parser.add_argument("--oracle-dir")
    parser.add_argument("--mmlu-prox-oracle-dir")
    parser.add_argument("--shard-dirs", nargs="+")
    parser.add_argument("--clean-combined-dir")
    parser.add_argument("--combined-dir")
    parser.add_argument("--clean-bank-version", default="wikipag-clean-57subjects-target1000-gpt54-v1")
    parser.add_argument("--combined-bank-version", default="global-mmlu-oracle-plus-wikipag-clean-57subjects-gpt54-v1")
    parser.add_argument("--embedding-model-dir", default="data/external/models/Qwen3-Embedding-4B")
    parser.add_argument("--embedding-device", default="cuda:1")
    parser.add_argument("--embedding-truncate-dim", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--poll-interval-s", type=int, default=60)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--clean-only", action="store_true", help="Finalize only the clean Wikipag shards.")
    args = parser.parse_args(argv)

    root = Path(args.root)
    args.oracle_dir = args.oracle_dir or (None if args.clean_only else str(root / "global_mmlu_oracle_cap1000_gpt54"))
    args.mmlu_prox_oracle_dir = args.mmlu_prox_oracle_dir or (
        None if args.clean_only else str(root / "mmlu_prox_non_global_oracle_cap1000_gpt54")
    )
    args.shard_dirs = args.shard_dirs or default_shard_dirs(root)
    args.clean_combined_dir = args.clean_combined_dir or str(root / "wikipag_clean_57subjects_target1000_gpt54_combined")
    args.combined_dir = args.combined_dir or (None if args.clean_only else str(root / "combined_oracle_plus_clean_57subjects_target1000_gpt54"))

    result = run(args)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
