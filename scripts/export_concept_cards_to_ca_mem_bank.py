from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, OrderedDict
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


def _take_texts(value: Any, *, limit: int | None = None) -> list[str]:
    items = value if isinstance(value, list) else ([] if value is None else [value])
    out: list[str] = []
    for item in items:
        text = _clean_text(item)
        if text and text not in out:
            out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def _extend_unique(target: list[str], value: Any, *, limit: int | None = None) -> None:
    if limit is not None and len(target) >= limit:
        return
    for text in _take_texts(value):
        if text not in target:
            target.append(text)
        if limit is not None and len(target) >= limit:
            return


def _load_concept_registry(source_dir: Path) -> dict[str, dict[str, Any]]:
    path = source_dir / "concept_registry.jsonl"
    if not path.exists():
        return {}
    registry: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        concept_id = _clean_text(row.get("concept_id"))
        if concept_id and concept_id not in registry:
            registry[concept_id] = row
    return registry


def _normalized_name(value: str) -> str:
    value = re.sub(r"[_/]+", " ", value.lower())
    value = re.sub(r"[^a-z0-9\s-]+", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _concept_key(row: dict[str, Any], registry: dict[str, dict[str, Any]], *, group_by_concept_name: bool) -> tuple[str, str]:
    subject = _clean_text(row.get("subject"))
    concept_id = _clean_text(row.get("concept_id"))
    if group_by_concept_name:
        registry_row = registry.get(concept_id, {})
        name = _clean_text(
            registry_row.get("normalized_name")
            or registry_row.get("canonical_name")
            or row.get("concept")
            or concept_id
        )
        return subject, _normalized_name(name) or concept_id
    if not concept_id:
        concept_id = stable_hash([subject, _normalized_name(_clean_text(row.get("concept")))], length=24)
    return subject, concept_id


def _is_active(row: dict[str, Any]) -> bool:
    return str(row.get("status", "")).lower() in {"", "active"}


def _usage_slot(row: dict[str, Any], *, runtime_unified: bool = False) -> dict[str, Any]:
    slot = {
        "usage_id": row.get("usage_id"),
        "usage_job_id": row.get("usage_job_id"),
        "usage_signature": _clean_text(row.get("usage_signature")) or None,
        "usage_pattern": _clean_text(row.get("usage_pattern")) or None,
        "concept_boundary": _clean_text(row.get("concept_boundary")) or None,
        "trigger_conditions": _take_texts(row.get("trigger_conditions")),
        "decision_procedure": _take_texts(row.get("decision_procedure")),
        "verification_rules": _take_texts(row.get("verification_rules")),
        "failure_boundaries": _take_texts(row.get("failure_boundaries")),
        "source_ids": _take_texts(row.get("source_ids")),
        "evidence_claim_count": len(row.get("evidence_claims") or []),
    }
    if runtime_unified:
        slot.pop("usage_id", None)
        slot.pop("usage_job_id", None)
        slot.pop("source_ids", None)
    return slot


def _slot_label(slot: dict[str, Any]) -> str:
    signature = _clean_text(slot.get("usage_signature"))
    pattern = _clean_text(slot.get("usage_pattern"))
    if signature and pattern:
        return f"{signature}: {pattern}"
    return signature or pattern


def _description(
    *,
    subject: str,
    concept_id: str,
    concept_name: str,
    registry_row: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    max_description_slots: int,
    runtime_unified: bool,
) -> str:
    definition = _clean_text((registry_row or {}).get("definition"))
    concept_type = _clean_text((registry_row or {}).get("concept_type"))
    aliases = _take_texts((registry_row or {}).get("aliases"), limit=8)
    slot_labels = [_slot_label(_usage_slot(row, runtime_unified=runtime_unified)) for row in rows]
    slot_labels = [text for text in dict.fromkeys(slot_labels) if text]

    boundaries: list[str] = []
    for row in rows:
        _extend_unique(boundaries, row.get("concept_boundary"), limit=4)

    parts = [
        "Concept-level CA-Mem card. Retrieval unit: one card per subject+concept.",
        f"Subject: {subject}.",
        f"Concept: {concept_name}.",
    ]
    if not runtime_unified:
        parts.append(f"Concept ID: {concept_id}.")
    if concept_type:
        parts.append(f"Concept type: {concept_type}.")
    if aliases:
        parts.append(f"Aliases: {', '.join(aliases)}.")
    if definition:
        parts.append(f"Definition: {definition}")
    if slot_labels:
        more = len(slot_labels) - max_description_slots
        parts.append(
            "Usage slots: "
            + " | ".join(slot_labels[:max_description_slots])
            + (f" | ... {more} more" if more > 0 else "")
        )
    if boundaries:
        parts.append("Boundaries: " + " | ".join(boundaries[:4]))
    return " ".join(part for part in parts if part)


def _node_from_group(
    *,
    subject: str,
    concept_id: str,
    rows: list[dict[str, Any]],
    registry: dict[str, dict[str, Any]],
    order: int,
    max_summary_triggers: int,
    max_summary_rules: int,
    max_summary_boundaries: int,
    max_description_slots: int,
    runtime_unified: bool,
) -> tuple[MemoryNode, dict[str, Any], dict[str, Any]]:
    registry_row = registry.get(concept_id)
    concept_name = _clean_text(
        (registry_row or {}).get("canonical_name")
        or rows[0].get("concept")
        or rows[0].get("concept_id")
        or concept_id
    )
    aliases = _take_texts((registry_row or {}).get("aliases"))
    definition = _clean_text((registry_row or {}).get("definition"))
    concept_type = _clean_text((registry_row or {}).get("concept_type"))

    triggers: list[str] = []
    rules: list[str] = []
    boundaries: list[str] = []
    source_ids: list[str] = []
    candidate_ids: list[str] = [concept_id]
    evidence_claim_count = 0
    usage_slots: list[dict[str, Any]] = []

    for row in rows:
        slot = _usage_slot(row, runtime_unified=runtime_unified)
        usage_slots.append(slot)
        evidence_claim_count += int(slot["evidence_claim_count"])
        _extend_unique(triggers, row.get("trigger_conditions"), limit=max_summary_triggers)
        if row.get("usage_signature"):
            _extend_unique(rules, f"Usage signature: {row.get('usage_signature')}", limit=max_summary_rules)
        if row.get("usage_pattern"):
            _extend_unique(rules, f"Usage pattern: {row.get('usage_pattern')}", limit=max_summary_rules)
        _extend_unique(rules, row.get("decision_procedure"), limit=max_summary_rules)
        _extend_unique(rules, row.get("verification_rules"), limit=max_summary_rules)
        _extend_unique(boundaries, row.get("failure_boundaries"), limit=max_summary_boundaries)
        _extend_unique(boundaries, row.get("concept_boundary"), limit=max_summary_boundaries)
        _extend_unique(source_ids, row.get("source_ids"), limit=64)
        _extend_unique(candidate_ids, [row.get("usage_id"), row.get("usage_job_id")], limit=64)

    public_concept_id = f"concept_{stable_hash([subject, concept_name], length=16)}"
    memory_id = (
        f"mem_concept_{stable_hash([subject, concept_name], length=16)}"
        if runtime_unified
        else f"mem_concept_{stable_hash([subject, concept_id], length=16)}"
    )
    node = MemoryNode.create(
        memory_id=memory_id,
        subject=subject,
        concept=concept_name,
        description=_description(
            subject=subject,
            concept_id=concept_id,
            concept_name=concept_name,
            registry_row=registry_row,
            rows=rows,
            max_description_slots=max_description_slots,
            runtime_unified=runtime_unified,
        ),
        usage_summary=UsageSummary(
            trigger_invariants=triggers[:max_summary_triggers],
            decision_rules=rules[:max_summary_rules],
            negative_error_boundaries=boundaries[:max_summary_boundaries],
        ),
        provenance=Provenance() if runtime_unified else Provenance(source_sample_ids=source_ids, candidate_ids=candidate_ids),
        order=order,
    )

    concept_card = {
        "memory_id": node.memory_id,
        "subject": subject,
        "concept_id": public_concept_id if runtime_unified else concept_id,
        "canonical_name": concept_name,
        "normalized_name": (registry_row or {}).get("normalized_name"),
        "definition": definition,
        "aliases": aliases,
        "concept_type": concept_type,
        "usage_slot_count": len(usage_slots),
        "usage_slots": usage_slots,
        "evidence_claim_count": evidence_claim_count,
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
        "concept": concept_name,
        "usage_slot_count": len(usage_slots),
        "usage_ids": [row.get("usage_id") for row in rows if row.get("usage_id")],
        "usage_job_ids": [row.get("usage_job_id") for row in rows if row.get("usage_job_id")],
        "source_ids": source_ids,
    }
    return node, concept_card, trace


def _runtime_trace(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_id": trace.get("memory_id"),
        "subject": trace.get("subject"),
        "concept_id": trace.get("runtime_concept_id"),
        "concept": trace.get("concept"),
        "usage_slot_count": trace.get("usage_slot_count"),
    }


def export_bank(
    source_dir: Path,
    output_dir: Path,
    *,
    max_cards: int | None = None,
    max_summary_triggers: int = 18,
    max_summary_rules: int = 30,
    max_summary_boundaries: int = 18,
    max_description_slots: int = 8,
    runtime_unified: bool = False,
    group_by_concept_name: bool = False,
) -> dict[str, Any]:
    source_cards_path = source_dir / "usage_cards.jsonl"
    if not source_cards_path.exists():
        raise FileNotFoundError(source_cards_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    registry = _load_concept_registry(source_dir)

    groups: OrderedDict[tuple[str, str], list[dict[str, Any]]] = OrderedDict()
    raw_usage_card_count = 0
    skipped_inactive_count = 0
    for row in read_jsonl(source_cards_path):
        if not _is_active(row):
            skipped_inactive_count += 1
            continue
        key = _concept_key(row, registry, group_by_concept_name=group_by_concept_name)
        groups.setdefault(key, []).append(row)
        raw_usage_card_count += 1
        if max_cards is not None and raw_usage_card_count >= max_cards:
            break

    nodes: list[MemoryNode] = []
    concept_cards: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    slot_counts: list[int] = []
    for order, ((subject, concept_id), rows) in enumerate(groups.items(), start=1):
        node, concept_card, trace = _node_from_group(
            subject=subject,
            concept_id=concept_id,
            rows=rows,
            registry=registry,
            order=order,
            max_summary_triggers=max_summary_triggers,
            max_summary_rules=max_summary_rules,
            max_summary_boundaries=max_summary_boundaries,
            max_description_slots=max_description_slots,
            runtime_unified=runtime_unified,
        )
        nodes.append(node)
        concept_cards.append(concept_card)
        traces.append(trace)
        slot_counts.append(len(rows))

    duplicate_memory_ids = [
        memory_id
        for memory_id, count in Counter(node.memory_id for node in nodes).items()
        if count > 1
    ]
    if duplicate_memory_ids:
        raise RuntimeError(f"duplicate generated memory_id values: {duplicate_memory_ids[:5]}")

    write_jsonl(output_dir / "bank.jsonl", [to_dict(node) for node in nodes])
    write_jsonl(output_dir / "concept_cards.jsonl", concept_cards)
    public_traces = [_runtime_trace(trace) for trace in traces] if runtime_unified else traces
    write_jsonl(output_dir / "concept_card_to_usage_trace.jsonl", public_traces)
    write_jsonl(output_dir / "card_to_memory_trace.jsonl", public_traces)
    if runtime_unified:
        write_jsonl(output_dir / "internal_audit_trace.jsonl", traces)
    write_jsonl(output_dir / "rejected.jsonl", [])

    embedder = HashingTextEmbedder()
    matrix = embedder.embed([key_for_node(node) for node in nodes])
    np.save(output_dir / "build_index.npy", matrix)

    manifest = {
        "builder": "wiki_concept_cards_to_ca_mem_adapter",
        "source_run_dir": str(source_dir),
        "source_usage_cards_path": str(source_cards_path),
        "source_concept_registry_path": str(source_dir / "concept_registry.jsonl"),
        "aggregation_key": ["subject", "concept_id"],
        "retrieval_unit": "concept_card",
        "one_memory_node_per_subject_concept": True,
        "runtime_unified_view": runtime_unified,
        "group_by_concept_name": group_by_concept_name,
        "raw_usage_card_count": raw_usage_card_count,
        "skipped_inactive_count": skipped_inactive_count,
        "concept_card_count": len(nodes),
        "usage_slot_count": raw_usage_card_count,
        "multi_slot_concept_card_count": sum(1 for value in slot_counts if value > 1),
        "max_usage_slots_per_concept_card": max(slot_counts) if slot_counts else 0,
        "avg_usage_slots_per_concept_card": (raw_usage_card_count / len(nodes)) if nodes else 0.0,
        "bank_version": len(nodes),
        "candidate_count": len(nodes),
        "final_node_count": len(nodes),
        "source_corpus_only": True,
        "final_bank_hash": bank_hash(nodes),
        "embedding_backend": "hash",
        "embedding_model": embedder.model_name,
        "build_index_path": str(output_dir / "build_index.npy"),
        "production_ready": False,
        "notes": [
            "One MemoryNode is produced per subject+concept.",
            "Multiple source usage rows are stored as usage_slots inside the concept card.",
            "Runtime-unified view hides provenance identifiers from public bank files when enabled.",
        ],
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
    parser = argparse.ArgumentParser(description="Aggregate wiki-clean usage_cards.jsonl into one CA-Mem concept card per subject+concept.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cards", type=int)
    parser.add_argument("--max-summary-triggers", type=int, default=18)
    parser.add_argument("--max-summary-rules", type=int, default=30)
    parser.add_argument("--max-summary-boundaries", type=int, default=18)
    parser.add_argument("--max-description-slots", type=int, default=8)
    parser.add_argument("--runtime-unified", action="store_true", help="Hide provenance identifiers from runtime bank and public concept cards.")
    parser.add_argument("--group-by-concept-name", action="store_true", help="Aggregate by subject + normalized concept name instead of source concept_id.")
    args = parser.parse_args(argv)
    manifest = export_bank(
        Path(args.source_dir),
        Path(args.output_dir),
        max_cards=args.max_cards,
        max_summary_triggers=args.max_summary_triggers,
        max_summary_rules=args.max_summary_rules,
        max_summary_boundaries=args.max_summary_boundaries,
        max_description_slots=args.max_description_slots,
        runtime_unified=args.runtime_unified,
        group_by_concept_name=args.group_by_concept_name,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
