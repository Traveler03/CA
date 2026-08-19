from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_subject_concept_smoke import run_pipeline
from scripts.export_concept_card_runtime_bank import write_runtime_bank
from src.smoke_test.io import read_jsonl, write_jsonl


DEFAULT_SUBJECTS = [
    "high_school_microeconomics",
    "high_school_macroeconomics",
    "high_school_biology",
    "conceptual_physics",
    "high_school_chemistry",
    "high_school_statistics",
    "formal_logic",
    "abstract_algebra",
    "machine_learning",
    "professional_law",
]

COMBINED_JSONL_FILES = [
    "concept_registry.jsonl",
    "concept_relations.jsonl",
    "usage_cards.jsonl",
    "usage_card_claims.jsonl",
    "rejected_items.jsonl",
    "build_events.jsonl",
    "usage_index.jsonl",
]


def load_subjects_from_global_mmlu(path: Path) -> list[str]:
    seen: set[str] = set()
    subjects: list[str] = []
    for row in read_jsonl(path):
        subject = str(row.get("subject") or "")
        if subject and subject not in seen:
            seen.add(subject)
            subjects.append(subject)
    return subjects


def select_subjects(args: argparse.Namespace) -> list[str]:
    if args.subjects:
        subjects = args.subjects
    elif args.preset == "bank-s":
        available = set(load_subjects_from_global_mmlu(Path(args.global_mmlu_en)))
        subjects = [subject for subject in DEFAULT_SUBJECTS if subject in available]
    else:
        subjects = load_subjects_from_global_mmlu(Path(args.global_mmlu_en))
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]
    return subjects


def build_subject_args(args: argparse.Namespace, subject: str, subject_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        subject=subject,
        category=None,
        global_mmlu_en=args.global_mmlu_en,
        output_dir=str(subject_dir),
        target_active_concepts=args.target_active_concepts,
        max_articles=args.max_articles,
        max_passages=args.max_passages,
        max_seed_queries=args.max_seed_queries,
        top_k_per_query=args.top_k_per_query,
        max_extraction_passages=args.max_extraction_passages,
        max_grounding_candidates=args.max_grounding_candidates,
        grounding_top_k=args.grounding_top_k,
        skip_llm_pair_judge=args.skip_llm_pair_judge,
        max_pair_judge_pairs=args.max_pair_judge_pairs,
        max_usage_concepts=args.max_usage_concepts,
        max_usage_jobs=args.max_usage_jobs,
        usage_retrieval_top_k=args.usage_retrieval_top_k,
        final_materials_per_usage_job=args.final_materials_per_usage_job,
        index_top_k=args.index_top_k,
        skip_faiss_usage_index=args.skip_faiss_usage_index,
        embedding_timeout_s=args.embedding_timeout_s,
        bank_version=args.bank_version,
        wikipag_service_url=args.wikipag_service_url,
        retrieval_timeout_s=args.retrieval_timeout_s,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        model_timeout_s=args.model_timeout_s,
    )


def copy_usage_index_shard(subject_dir: Path, bank_dir: Path, subject: str) -> None:
    src_dir = subject_dir / "usage_indexes" / subject
    if not src_dir.exists():
        return
    dst_dir = bank_dir / "usage_indexes" / subject
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, dst_dir / path.name)


def combine_outputs(bank_dir: Path, subject_dirs: list[tuple[str, Path]]) -> dict[str, Any]:
    combined_counts: dict[str, int] = {}
    for file_name in COMBINED_JSONL_FILES:
        rows: list[dict[str, Any]] = []
        for _subject, subject_dir in subject_dirs:
            path = subject_dir / file_name
            if path.exists():
                rows.extend(read_jsonl(path))
        write_jsonl(bank_dir / file_name, rows)
        combined_counts[file_name] = len(rows)

    try:
        import pandas as pd

        frames = []
        for _subject, subject_dir in subject_dirs:
            path = subject_dir / "evidence_sections.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
        if frames:
            pd.concat(frames, ignore_index=True).drop_duplicates(subset=["source_id"]).to_parquet(
                bank_dir / "evidence_sections.parquet",
                index=False,
            )
    except Exception as exc:
        (bank_dir / "evidence_sections.error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")

    for subject, subject_dir in subject_dirs:
        copy_usage_index_shard(subject_dir, bank_dir, subject)

    return combined_counts


def build_bank_manifest(
    args: argparse.Namespace,
    *,
    subjects: list[str],
    subject_summaries: list[dict[str, Any]],
    combined_counts: dict[str, int],
    runtime_bank: dict[str, Any] | None = None,
) -> dict[str, Any]:
    totals = Counter()
    for summary in subject_summaries:
        for key in [
            "active_concept_count",
            "concept_count",
            "active_card_count",
            "usage_claim_count",
            "usage_index_count",
            "usage_faiss_index_count",
            "evidence_section_count",
            "rejected_item_count",
            "model_network_calls",
        ]:
            totals[key] += int(summary.get(key) or 0)
    return {
        "bank_version": args.bank_version,
        "preset": args.preset,
        "source_snapshot": "wikipedia-en-2026-07-01-sherlock-faiss",
        "subjects": len(subjects),
        "subject_ids": subjects,
        "concept_count": totals["concept_count"],
        "active_concept_count": totals["active_concept_count"],
        "active_card_count": totals["active_card_count"],
        "usage_claim_count": totals["usage_claim_count"],
        "usage_index_count": totals["usage_index_count"],
        "usage_faiss_index_count": totals["usage_faiss_index_count"],
        "evidence_section_count": totals["evidence_section_count"],
        "rejected_item_count": totals["rejected_item_count"],
        "model_network_calls": totals["model_network_calls"],
        "construction_model": args.model,
        "embedding_model": "Qwen3-Embedding-4B for local Wikipag retrieval",
        "top_k": args.index_top_k,
        "similarity_gate": False,
        "construction_cutoff": datetime.now(timezone.utc).isoformat(),
        "benchmark_content_accessed": False,
        "combined_counts": combined_counts,
        "runtime_bank": runtime_bank,
        "shard_root": "subjects",
    }


async def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    bank_dir = Path(args.output_dir)
    bank_dir.mkdir(parents=True, exist_ok=True)
    subjects = select_subjects(args)
    write_jsonl(
        bank_dir / "subject_queue.jsonl",
        [{"subject": subject, "status": "queued", "ordinal": idx} for idx, subject in enumerate(subjects)],
    )

    subject_dirs: list[tuple[str, Path]] = []
    subject_summaries: list[dict[str, Any]] = []
    for idx, subject in enumerate(subjects):
        subject_dir = bank_dir / "subjects" / subject
        summary_path = subject_dir / "summary.json"
        if args.resume and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            subject_args = build_subject_args(args, subject, subject_dir)
            summary = await run_pipeline(subject_args)
        subject_dirs.append((subject, subject_dir))
        subject_summaries.append(summary)
        write_jsonl(
            bank_dir / "subject_queue.jsonl",
            [
                {
                    "subject": item,
                    "status": "done" if pos <= idx else "queued",
                    "ordinal": pos,
                }
                for pos, item in enumerate(subjects)
            ],
        )

    combined_counts = combine_outputs(bank_dir, subject_dirs)
    runtime_bank = None
    if args.write_runtime_bank:
        runtime_bank = write_runtime_bank(
            input_dir=bank_dir,
            build_embedding_index=bool(args.build_runtime_embedding_index),
            embedding_model_dir=Path(args.embedding_model_dir),
            embedding_device=args.embedding_device,
            embedding_truncate_dim=int(args.embedding_truncate_dim),
            embedding_batch_size=int(args.embedding_batch_size),
        )
    manifest = build_bank_manifest(
        args,
        subjects=subjects,
        subject_summaries=subject_summaries,
        combined_counts=combined_counts,
        runtime_bank=runtime_bank,
    )
    (bank_dir / "bank_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (bank_dir / "summary.json").write_text(
        json.dumps(
            {
                "subject": "batch",
                "subjects": len(subjects),
                "model": args.model,
                "active_concept_count": manifest["active_concept_count"],
                "concept_count": manifest["concept_count"],
                "active_card_count": manifest["active_card_count"],
                "usage_claim_count": manifest["usage_claim_count"],
                "usage_index_count": manifest["usage_index_count"],
                "usage_faiss_index_count": manifest["usage_faiss_index_count"],
                "model_network_calls": manifest["model_network_calls"],
                "benchmark_content_accessed": False,
                "runtime_bank": runtime_bank,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (bank_dir / "subject_summaries.json").write_text(
        json.dumps(subject_summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a sequential subject queue for Usage Bank construction.")
    parser.add_argument("--output-dir", default="runs/smoke_001/usage_bank_batch_smoke_gpt54")
    parser.add_argument("--global-mmlu-en", default="data/processed/global_mmlu/en.jsonl")
    parser.add_argument("--preset", choices=["bank-s", "all"], default="bank-s")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--target-active-concepts", type=int, default=20)
    parser.add_argument("--max-articles", type=int, default=30)
    parser.add_argument("--max-passages", type=int, default=24)
    parser.add_argument("--max-seed-queries", type=int, default=8)
    parser.add_argument("--top-k-per-query", type=int, default=8)
    parser.add_argument("--max-extraction-passages", type=int, default=8)
    parser.add_argument("--max-grounding-candidates", type=int, default=30)
    parser.add_argument("--grounding-top-k", type=int, default=3)
    parser.add_argument("--skip-llm-pair-judge", action="store_true")
    parser.add_argument("--max-pair-judge-pairs", type=int, default=40)
    parser.add_argument("--max-usage-concepts", type=int, default=5)
    parser.add_argument("--max-usage-jobs", type=int, default=10)
    parser.add_argument("--usage-retrieval-top-k", type=int, default=8)
    parser.add_argument("--final-materials-per-usage-job", type=int, default=4)
    parser.add_argument("--index-top-k", type=int, default=3)
    parser.add_argument("--skip-faiss-usage-index", action="store_true")
    parser.add_argument("--embedding-timeout-s", type=float, default=120.0)
    parser.add_argument("--write-runtime-bank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-runtime-embedding-index", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embedding-model-dir", type=Path, default=Path("data/external/models/Qwen3-Embedding-4B"))
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-truncate-dim", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--bank-version", default="bank-smoke-gpt54-v0.1")
    parser.add_argument("--wikipag-service-url", default="http://127.0.0.1:8897")
    parser.add_argument("--retrieval-timeout-s", type=float, default=120.0)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--model-timeout-s", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    manifest = asyncio.run(run_batch(args))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
