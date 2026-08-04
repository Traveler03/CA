from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from src.utils.hash import stable_hash


@dataclass(frozen=True)
class Provenance:
    source_sample_ids: list[str] = field(default_factory=list)
    candidate_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "Provenance":
        value = value or {}
        return cls(
            source_sample_ids=[str(item) for item in value.get("source_sample_ids") or []],
            candidate_ids=[str(item) for item in value.get("candidate_ids") or []],
        )


@dataclass(frozen=True)
class UsageSummary:
    trigger_invariants: list[str] = field(default_factory=list)
    decision_rules: list[str] = field(default_factory=list)
    negative_error_boundaries: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "UsageSummary":
        value = value or {}
        return cls(
            trigger_invariants=coerce_sentence_list(value.get("trigger_invariants")),
            decision_rules=coerce_sentence_list(value.get("decision_rules")),
            negative_error_boundaries=coerce_sentence_list(value.get("negative_error_boundaries")),
        )


@dataclass(frozen=True)
class MemoryNode:
    memory_id: str
    version: int
    status: str
    subject: str
    concept: str
    description: str
    usage_summary: UsageSummary
    provenance: Provenance
    created_at_order: int
    last_updated_at_order: int
    content_hash: str
    update_count: int = 0

    @classmethod
    def create(
        cls,
        *,
        memory_id: str,
        subject: str,
        concept: str,
        description: str,
        usage_summary: UsageSummary,
        provenance: Provenance,
        order: int,
    ) -> "MemoryNode":
        node = cls(
            memory_id=memory_id,
            version=1,
            status="active",
            subject=subject,
            concept=concept,
            description=description,
            usage_summary=usage_summary,
            provenance=provenance,
            created_at_order=order,
            last_updated_at_order=order,
            content_hash="",
            update_count=0,
        )
        return node.with_content_hash()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryNode":
        return cls(
            memory_id=str(value["memory_id"]),
            version=int(value.get("version", 1)),
            status=str(value.get("status", "active")),
            subject=str(value["subject"]),
            concept=str(value["concept"]),
            description=str(value["description"]),
            usage_summary=UsageSummary.from_dict(value.get("usage_summary")),
            provenance=Provenance.from_dict(value.get("provenance")),
            created_at_order=int(value.get("created_at_order", 0)),
            last_updated_at_order=int(value.get("last_updated_at_order", 0)),
            content_hash=str(value.get("content_hash", "")),
            update_count=int(value.get("update_count", 0)),
        )

    def with_content_hash(self) -> "MemoryNode":
        return replace(self, content_hash=memory_content_hash(self))


def coerce_sentence_list(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else str(value).replace(";", "\n").split("\n")
    normalized: list[str] = []
    for item in items:
        text = " ".join(str(item).strip().split())
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)


def memory_content_payload(node: MemoryNode) -> dict[str, Any]:
    return {
        "status": node.status,
        "subject": node.subject,
        "concept": node.concept,
        "description": node.description,
        "usage_summary": asdict(node.usage_summary),
    }


def memory_content_hash(node: MemoryNode) -> str:
    return stable_hash(memory_content_payload(node), length=24)


def bank_hash(nodes: list[MemoryNode]) -> str:
    payload = [to_dict(node) for node in sorted(nodes, key=lambda item: item.memory_id)]
    return stable_hash(payload, length=32)
