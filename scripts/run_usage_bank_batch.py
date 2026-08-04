from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_subject_concept_smoke import run_pipeline
from src.ca_mem.embedding import HashingTextEmbedder
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
    "subject_profile.jsonl",
    "concept_queries.jsonl",
    "concept_passages.jsonl",
    "candidate_concepts.jsonl",
    "concept_registry.jsonl",
    "concept_evidence.jsonl",
    "merge_redirects.jsonl",
    "evidence_packs.jsonl",
    "runtime_cards.raw.jsonl",
    "runtime_card_claims.jsonl",
    "runtime_cards.jsonl",
    "runtime_card_index.jsonl",
    "build_events.jsonl",
    "rejected_items.jsonl",
]


def load_subjects_from_subject_source(path: Path) -> list[str]:
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
        available = set(load_subjects_from_subject_source(Path(args.subject_source_jsonl)))
        subjects = [subject for subject in DEFAULT_SUBJECTS if subject in available]
    else:
        subjects = load_subjects_from_subject_source(Path(args.subject_source_jsonl))
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]
    return subjects


def build_subject_args(args: argparse.Namespace, subject: str, subject_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        subject=subject,
        category=None,
        subject_source_jsonl=args.subject_source_jsonl,
        output_dir=str(subject_dir),
        target_active_concepts=args.target_active_concepts,
        max_topic_anchors=args.max_topic_anchors,
        max_concept_queries=args.max_concept_queries,
        max_passages=args.max_passages,
        top_k_per_query=args.top_k_per_query,
        max_extraction_passages=args.max_extraction_passages,
        max_grounding_candidates=args.max_grounding_candidates,
        grounding_top_k=args.grounding_top_k,
        evidence_top_k=args.evidence_top_k,
        max_evidence_items_per_slot=args.max_evidence_items_per_slot,
        max_card_concepts=args.max_card_concepts,
        bank_version=args.bank_version,
        wikipag_service_url=args.wikipag_service_url,
        retrieval_timeout_s=args.retrieval_timeout_s,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        model_timeout_s=args.model_timeout_s,
    )


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

    index_path = bank_dir / "runtime_card_index.jsonl"
    if index_path.exists():
        rows = list(read_jsonl(index_path))
        embedder = HashingTextEmbedder()
        matrix = embedder.embed([str(row.get("index_text") or "") for row in rows])
        import numpy as np

        np.save(bank_dir / "runtime_card_index.npy", matrix)
        (bank_dir / "runtime_card_index_meta.json").write_text(
            json.dumps(
                {
                    "index_type": "numpy_dense_matrix",
                    "metric": "cosine_on_normalized_embeddings",
                    "embedding_backend": "hash",
                    "embedding_model": embedder.model_name,
                    "dimension": int(matrix.shape[1]) if matrix.ndim == 2 else embedder.dim,
                    "count": len(rows),
                    "matrix_path": str(bank_dir / "runtime_card_index.npy"),
                    "ids_path": str(index_path),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return combined_counts


def build_bank_manifest(
    args: argparse.Namespace,
    *,
    subjects: list[str],
    subject_summaries: list[dict[str, Any]],
    combined_counts: dict[str, int],
) -> dict[str, Any]:
    totals = Counter()
    for summary in subject_summaries:
        for key in [
            "active_concept_count",
            "concept_count",
            "active_card_count",
            "runtime_card_count",
            "runtime_card_claim_count",
            "runtime_card_index_count",
            "evidence_pack_count",
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
        "runtime_card_count": combined_counts.get("runtime_cards.jsonl", 0),
        "runtime_card_claim_count": combined_counts.get("runtime_card_claims.jsonl", 0),
        "runtime_card_index_count": combined_counts.get("runtime_card_index.jsonl", 0),
        "evidence_pack_count": combined_counts.get("evidence_packs.jsonl", 0),
        "evidence_section_count": totals["evidence_section_count"],
        "rejected_item_count": totals["rejected_item_count"],
        "model_network_calls": totals["model_network_calls"],
        "construction_model": args.model,
        "pipeline_version": "wiki_compact_card_v1",
        "embedding_model": "deterministic-hashing-embedding for runtime card index",
        "construction_cutoff": datetime.now(timezone.utc).isoformat(),
        "source_corpus_only": True,
        "one_card_per_subject_concept": True,
        "combined_counts": combined_counts,
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
    manifest = build_bank_manifest(args, subjects=subjects, subject_summaries=subject_summaries, combined_counts=combined_counts)
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
                "runtime_card_count": manifest["runtime_card_count"],
                "runtime_card_claim_count": manifest["runtime_card_claim_count"],
                "runtime_card_index_count": manifest["runtime_card_index_count"],
                "model_network_calls": manifest["model_network_calls"],
                "source_corpus_only": True,
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
    parser = argparse.ArgumentParser(description="Run a sequential subject queue for Wiki compact-card construction.")
    parser.add_argument("--output-dir", default="runs/smoke_001/wiki_compact_card_batch_smoke")
    parser.add_argument("--subject-source-jsonl", default="data/subject_source/en_subjects.jsonl")
    parser.add_argument("--preset", choices=["bank-s", "all"], default="bank-s")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--target-active-concepts", type=int, default=20)
    parser.add_argument("--max-topic-anchors", type=int, default=12)
    parser.add_argument("--max-concept-queries", type=int, default=24)
    parser.add_argument("--max-passages", type=int, default=32)
    parser.add_argument("--top-k-per-query", type=int, default=8)
    parser.add_argument("--max-extraction-passages", type=int, default=10)
    parser.add_argument("--max-grounding-candidates", type=int, default=40)
    parser.add_argument("--grounding-top-k", type=int, default=4)
    parser.add_argument("--evidence-top-k", type=int, default=6)
    parser.add_argument("--max-evidence-items-per-slot", type=int, default=2)
    parser.add_argument("--max-card-concepts", type=int, default=10)
    parser.add_argument("--bank-version", default="wiki-compact-card-batch-v0.1")
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
