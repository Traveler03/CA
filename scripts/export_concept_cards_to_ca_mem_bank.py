from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ca_mem.embedding import HashingTextEmbedder, key_for_node
from src.ca_mem.schemas import MemoryNode, Provenance, UsageSummary, bank_hash, to_dict
from src.utils.hash import stable_hash
from src.utils.jsonl import read_jsonl, write_jsonl


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _take_texts(value: Any) -> list[str]:
    items = value if isinstance(value, list) else ([] if value is None else [value])
    out: list[str] = []
    for item in items:
        text = _clean_text(item)
        if text and text not in out:
            out.append(text)
    return out


def _description(card: dict[str, Any], *, runtime_unified: bool) -> str:
    parts = [
        "Compact concept card. Retrieval unit: one card per subject+concept.",
        f"Subject: {_clean_text(card.get('subject'))}.",
        f"Concept: {_clean_text(card.get('concept'))}.",
    ]
    if not runtime_unified:
        parts.append(f"Concept ID: {_clean_text(card.get('concept_id'))}.")
    if card.get("definition"):
        parts.append(f"Definition: {_clean_text(card.get('definition'))}")
    trigger = _take_texts(card.get("trigger"))
    rule = _take_texts(card.get("rule"))
    pitfall = _take_texts(card.get("pitfall"))
    if trigger:
        parts.append("Trigger: " + " | ".join(trigger))
    if rule:
        parts.append("Rule: " + " | ".join(rule))
    if pitfall:
        parts.append("Pitfall: " + " | ".join(pitfall))
    return " ".join(part for part in parts if part)


def _node_from_card(card: dict[str, Any], *, order: int, runtime_unified: bool) -> tuple[MemoryNode, dict[str, Any], dict[str, Any]]:
    subject = _clean_text(card.get("subject"))
    concept = _clean_text(card.get("concept"))
    concept_id = _clean_text(card.get("concept_id")) or stable_hash([subject, concept], length=16)
    public_concept_id = f"concept_{stable_hash([subject, concept], length=16)}"
    memory_id = (
        f"mem_concept_{stable_hash([subject, concept], length=16)}"
        if runtime_unified
        else f"mem_concept_{stable_hash([subject, concept_id], length=16)}"
    )
    trigger = _take_texts(card.get("trigger"))
    rule = _take_texts(card.get("rule"))
    pitfall = _take_texts(card.get("pitfall"))
    source_ids = _take_texts(card.get("source_ids"))

    node = MemoryNode.create(
        memory_id=memory_id,
        subject=subject,
        concept=concept,
        description=_description(card, runtime_unified=runtime_unified),
        usage_summary=UsageSummary(
            trigger_invariants=trigger,
            decision_rules=rule,
            negative_error_boundaries=pitfall,
        ),
        provenance=Provenance() if runtime_unified else Provenance(source_sample_ids=source_ids, candidate_ids=[_clean_text(card.get("card_id"))]),
        order=order,
    )
    concept_card = {
        "memory_id": node.memory_id,
        "subject": subject,
        "concept_id": public_concept_id if runtime_unified else concept_id,
        "canonical_name": concept,
        "definition": _clean_text(card.get("definition")),
        "trigger": trigger,
        "rule": rule,
        "pitfall": pitfall,
        "status": "active",
        "version": 1,
    }
    if not runtime_unified:
        concept_card["source_ids"] = source_ids
    trace = {
        "memory_id": node.memory_id,
        "subject": subject,
        "concept_id": concept_id,
        "runtime_concept_id": public_concept_id if runtime_unified else concept_id,
        "concept": concept,
        "card_id": card.get("card_id"),
        "source_ids": source_ids,
    }
    return node, concept_card, trace


def export_bank(
    source_dir: Path,
    output_dir: Path,
    *,
    max_cards: int | None = None,
    runtime_unified: bool = False,
) -> dict[str, Any]:
    source_cards_path = source_dir / "runtime_cards.jsonl"
    if not source_cards_path.exists():
        raise FileNotFoundError(source_cards_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_cards = [row for row in read_jsonl(source_cards_path) if str(row.get("status", "active")).lower() == "active"]
    if max_cards is not None:
        raw_cards = raw_cards[:max_cards]

    seen: set[tuple[str, str]] = set()
    nodes: list[MemoryNode] = []
    concept_cards: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for row in raw_cards:
        key = (_clean_text(row.get("subject")), _clean_text(row.get("concept_id")))
        if key in seen:
            raise RuntimeError(f"duplicate runtime card for subject+concept: {key}")
        seen.add(key)
        node, concept_card, trace = _node_from_card(row, order=len(nodes) + 1, runtime_unified=runtime_unified)
        nodes.append(node)
        concept_cards.append(concept_card)
        traces.append(trace)

    write_jsonl(output_dir / "bank.jsonl", [to_dict(node) for node in nodes])
    write_jsonl(output_dir / "concept_cards.jsonl", concept_cards)
    public_traces = [
        {
            "memory_id": trace["memory_id"],
            "subject": trace["subject"],
            "concept_id": trace["runtime_concept_id"],
            "concept": trace["concept"],
        }
        for trace in traces
    ] if runtime_unified else traces
    write_jsonl(output_dir / "concept_card_to_usage_trace.jsonl", public_traces)
    write_jsonl(output_dir / "card_to_memory_trace.jsonl", public_traces)
    if runtime_unified:
        write_jsonl(output_dir / "internal_audit_trace.jsonl", traces)
    write_jsonl(output_dir / "rejected.jsonl", [])

    embedder = HashingTextEmbedder()
    matrix = embedder.embed([key_for_node(node) for node in nodes])
    np.save(output_dir / "build_index.npy", matrix)

    manifest = {
        "builder": "runtime_cards_to_ca_mem_adapter",
        "source_run_dir": str(source_dir),
        "source_runtime_cards_path": str(source_cards_path),
        "aggregation_key": ["subject", "concept_id"],
        "retrieval_unit": "compact_concept_card",
        "one_memory_node_per_subject_concept": True,
        "runtime_unified_view": runtime_unified,
        "source_card_count": len(raw_cards),
        "concept_card_count": len(nodes),
        "bank_version": len(nodes),
        "candidate_count": len(nodes),
        "final_node_count": len(nodes),
        "source_corpus_only": True,
        "final_bank_hash": bank_hash(nodes),
        "embedding_backend": "hash",
        "embedding_model": embedder.model_name,
        "build_index_path": str(output_dir / "build_index.npy"),
        "production_ready": False,
    }
    (output_dir / "bank_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    snapshots = output_dir / "snapshots"
    snapshots.mkdir(exist_ok=True)
    shutil.copy2(output_dir / "bank.jsonl", snapshots / f"bank_v{len(nodes):06d}.jsonl")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export final runtime_cards.jsonl into a CA-Mem bank.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cards", type=int)
    parser.add_argument("--runtime-unified", action="store_true", help="Hide provenance identifiers from public bank files.")
    args = parser.parse_args(argv)
    manifest = export_bank(
        Path(args.source_dir),
        Path(args.output_dir),
        max_cards=args.max_cards,
        runtime_unified=args.runtime_unified,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
