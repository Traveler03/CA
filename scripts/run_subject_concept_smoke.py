from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.smoke_test.config import ModelConfig
from src.smoke_test.io import read_jsonl, write_jsonl
from src.smoke_test.model_client import SmokeModelClient
from src.smoke_test.stage_utils import stable_id


DEFAULT_SUBJECT = "high_school_microeconomics"
DEFAULT_CATEGORY = "social_sciences"
BAD_SECTION_TERMS = {
    "references",
    "external links",
    "bibliography",
    "further reading",
    "see also",
    "notes",
    "sources",
}
TOO_BROAD_NAMES = {
    "economics",
    "microeconomics",
    "macroeconomics",
    "mathematics",
    "science",
    "market",
    "business",
    "consumer",
    "producer",
    "theory",
    "model",
}
USAGE_WORDS = {
    "choosing",
    "selecting",
    "calculating",
    "using",
    "applying",
    "solving",
    "determining",
}
CURATED_SUBJECT_SEEDS: dict[str, list[str]] = {
    "high_school_microeconomics": [
        "microeconomics concepts",
        "consumer theory economics",
        "producer theory economics",
        "supply and demand economics",
        "market equilibrium economics",
        "price elasticity of demand",
        "opportunity cost economics",
        "production cost economics",
        "market structure economics",
        "externality economics",
        "marginal cost marginal benefit",
        "comparative advantage economics",
    ],
    "high_school_macroeconomics": [
        "macroeconomics concepts",
        "gross domestic product",
        "inflation unemployment",
        "aggregate demand aggregate supply",
        "monetary policy fiscal policy",
        "business cycle economics",
    ],
    "high_school_biology": [
        "biology concepts",
        "cell biology",
        "genetics concepts",
        "evolution biology",
        "photosynthesis cellular respiration",
    ],
}


class SubjectPlan(BaseModel):
    subject_id: str
    category: str
    target_active_concepts: int
    max_articles: int
    max_passages: int
    status: str = "planned"


class SeedQueryPlan(BaseModel):
    subject: str
    seed_queries: list[str]


class RetrievedMaterial(BaseModel):
    source_id: str
    subject: str
    title: str | None = None
    section: str | None = None
    passage_id: str
    text: str
    retrieval_query: str
    retrieval_score: float
    material_score: float
    rank: int


class CandidateDraft(BaseModel):
    raw_name: str
    proposed_name: str
    concept_type: str
    definition_candidate: str
    evidence_span: str
    aliases: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ExtractionPayload(BaseModel):
    candidates: list[CandidateDraft] = Field(default_factory=list)


class RawCandidate(BaseModel):
    candidate_id: str
    subject: str
    raw_name: str
    proposed_name: str
    normalized_name: str
    concept_type: str
    definition_candidate: str
    source_id: str
    passage_id: str
    title: str | None = None
    section: str | None = None
    evidence_span: str
    aliases: list[str] = Field(default_factory=list)
    confidence: float
    extraction_error: str | None = None


class FilteredCandidate(RawCandidate):
    filter_status: str
    filter_reasons: list[str] = Field(default_factory=list)


class GroundingEvidence(BaseModel):
    source_id: str
    passage_id: str
    title: str | None = None
    section: str | None = None
    evidence_span: str
    retrieval_query: str | None = None
    retrieval_score: float | None = None


class GroundedConcept(BaseModel):
    candidate_id: str
    subject: str
    grounding_status: str
    canonical_name_suggestion: str
    normalized_name: str
    definition: str
    scope: str
    concept_type: str
    aliases: list[str] = Field(default_factory=list)
    evidence: list[GroundingEvidence] = Field(default_factory=list)
    confidence: float


class CandidatePair(BaseModel):
    candidate_a: str
    candidate_b: str
    relation: str
    confidence: float
    blocking_reasons: list[str]
    evidence_source_ids: list[str] = Field(default_factory=list)


class PairJudgementPayload(BaseModel):
    relation: str
    confidence: float
    rationale: str = ""


class ConceptRegistryRow(BaseModel):
    concept_id: str
    subject_scopes: list[str]
    canonical_name: str
    normalized_name: str
    aliases: list[str] = Field(default_factory=list)
    definition: str
    concept_type: str
    status: str
    version: int = 1


class ConceptEvidenceRow(BaseModel):
    concept_id: str
    source_id: str
    passage_id: str
    title: str | None = None
    section: str | None = None
    evidence_span: str
    evidence_type: str = "definition"


class ConceptRelationRow(BaseModel):
    source_concept_id: str
    relation: str
    target_concept_id: str
    evidence_source_ids: list[str] = Field(default_factory=list)
    confidence: float


class MergeRedirectRow(BaseModel):
    from_candidate_id: str
    to_concept_id: str
    relation: str = "SAME"
    confidence: float


class ConceptQualityRow(BaseModel):
    concept_id: str
    evidence_support: int
    granularity: int
    subject_relevance: int
    canonicalization: int
    duplicate_risk: int
    status: str


class ConceptIndexRow(BaseModel):
    concept_id: str
    index_text: str


class UsageJobRow(BaseModel):
    usage_job_id: str
    subject: str
    concept_id: str
    concept: str
    job_type: str
    retrieval_query: str
    confusable_concept_id: str | None = None
    confusable_concept: str | None = None


class UsageMaterialRow(BaseModel):
    usage_job_id: str
    source_id: str
    subject: str
    concept_id: str
    concept: str
    job_type: str
    title: str | None = None
    section: str | None = None
    passage_id: str
    text: str
    retrieval_query: str
    retrieval_score: float
    concept_relevance: int
    usage_relevance: int
    supports_rule: bool
    supports_boundary: bool
    generic_definition_only: bool
    decision: str
    rank: int


class EvidenceClaimDraft(BaseModel):
    field: str
    claim: str
    source_ids: list[str] = Field(default_factory=list)
    support_type: str = "direct"


class UsageCardDraft(BaseModel):
    usage_pattern: str
    usage_signature: str
    concept_boundary: str
    trigger_conditions: list[str] = Field(default_factory=list)
    decision_procedure: list[str] = Field(default_factory=list)
    failure_boundaries: list[str] = Field(default_factory=list)
    verification_rules: list[str] = Field(default_factory=list)
    evidence_claims: list[EvidenceClaimDraft] = Field(default_factory=list)


class UsageExtractionPayload(BaseModel):
    cards: list[UsageCardDraft] = Field(default_factory=list)


class UsageCardRow(BaseModel):
    usage_id: str
    usage_job_id: str
    subject: str
    concept_id: str
    concept: str
    usage_pattern: str
    usage_signature: str
    concept_boundary: str
    trigger_conditions: list[str] = Field(default_factory=list)
    decision_procedure: list[str] = Field(default_factory=list)
    failure_boundaries: list[str] = Field(default_factory=list)
    verification_rules: list[str] = Field(default_factory=list)
    evidence_claims: list[EvidenceClaimDraft] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    version: int = 1
    status: str = "provisional"


class UsageCardClaimRow(BaseModel):
    claim_id: str
    usage_id: str
    usage_job_id: str
    concept_id: str
    field: str
    claim: str
    supporting_source_ids: list[str] = Field(default_factory=list)
    support_type: str
    contradiction: bool = False
    decision: str


class ConsolidationRow(BaseModel):
    usage_id: str
    subject: str
    concept_id: str
    before: dict[str, Any] | None = None
    candidate: dict[str, Any]
    decision: str
    patch: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] | None = None
    source_provenance: list[str] = Field(default_factory=list)


class UsageIndexRow(BaseModel):
    usage_id: str
    subject: str
    concept_id: str
    index_key: str
    payload: str
    status: str


class UsageIndexVectorMeta(BaseModel):
    subject: str
    index_type: str
    metric: str
    dimension: int
    count: int
    embedding_endpoint: str
    embedding_input: str
    faiss_path: str
    ids_path: str


class BuildEventRow(BaseModel):
    event_id: str
    stage: str
    item_id: str
    status: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class RejectedItemRow(BaseModel):
    item_id: str
    item_type: str
    stage: str
    reason: str
    payload: dict[str, Any] = Field(default_factory=dict)


def normalize_concept_name(value: str) -> str:
    text = value.lower().strip()
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"[‐‑‒–—-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]+", "", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if (
        len(words) > 1
        and words[-1].endswith("s")
        and not words[-1].endswith(("ss", "sis", "us", "ics"))
    ):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def subject_label(subject_id: str) -> str:
    return subject_id.replace("_", " ").strip()


def subject_prefix(subject_id: str) -> str:
    tokens = [tok for tok in subject_id.split("_") if tok not in {"high", "school", "college", "professional"}]
    short = "".join(tok[:4].upper() for tok in tokens[:2])
    return short or "CONCEPT"


def slug_name(value: str) -> str:
    text = normalize_concept_name(value)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text.upper()[:80] or "UNNAMED"


def load_subject_category(data_path: Path, subject: str) -> str:
    if not data_path.exists():
        return DEFAULT_CATEGORY
    for row in read_jsonl(data_path):
        if row.get("subject") == subject:
            raw = str(row.get("subject_category") or row.get("category") or DEFAULT_CATEGORY)
            return raw.lower().replace(" ", "_")
    return DEFAULT_CATEGORY


def build_subject_plan(args: argparse.Namespace) -> SubjectPlan:
    category = args.category or load_subject_category(Path(args.subject_source_jsonl), args.subject)
    return SubjectPlan(
        subject_id=args.subject,
        category=category,
        target_active_concepts=args.target_active_concepts,
        max_articles=args.max_articles,
        max_passages=args.max_passages,
    )


def generate_seed_queries(subject: str, *, limit: int) -> SeedQueryPlan:
    label = subject_label(subject)
    seeds = [
        label,
        f"{label} concepts",
        f"{label} definitions",
        f"{label} principles",
        f"{label} theory",
    ]
    seeds.extend(CURATED_SUBJECT_SEEDS.get(subject, []))
    seen: set[str] = set()
    deduped: list[str] = []
    for seed in seeds:
        cleaned = re.sub(r"\s+", " ", seed).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            deduped.append(cleaned)
        if len(deduped) >= limit:
            break
    return SeedQueryPlan(subject=subject, seed_queries=deduped)


def material_quality_score(text: str, title: str | None, section: str | None, subject: str) -> float:
    lowered = text.lower()
    score = 0.0
    definition_patterns = [
        " is a ",
        " is an ",
        " refers to ",
        " is defined as ",
        " describes ",
        " concept ",
        " principle ",
        " theory ",
        " law of ",
        " in economics",
        " in microeconomics",
    ]
    score += sum(1.0 for pattern in definition_patterns if pattern in lowered)
    subject_terms = [tok for tok in subject_label(subject).split() if tok not in {"high", "school"}]
    score += sum(0.7 for term in subject_terms if term in lowered)
    econ_terms = [
        "demand",
        "supply",
        "market",
        "price",
        "cost",
        "elasticity",
        "consumer",
        "producer",
        "firm",
        "goods",
        "utility",
        "marginal",
        "equilibrium",
        "externality",
    ]
    score += min(4.0, sum(0.35 for term in econ_terms if term in lowered))
    if title and any(term in title.lower() for term in econ_terms):
        score += 1.0
    if section and section.lower() in {"lead", "introduction", "overview"}:
        score += 0.3
    return score


def should_keep_material(hit: dict[str, Any], subject: str) -> bool:
    if hit.get("missing"):
        return False
    text = str(hit.get("text") or "").strip()
    if len(text) < 240:
        return False
    title = str(hit.get("title") or "")
    section = str(hit.get("section") or "")
    section_l = section.lower()
    if any(term == section_l or term in section_l for term in BAD_SECTION_TERMS):
        return False
    if title.lower().endswith("(disambiguation)"):
        return False
    return material_quality_score(text, title, section, subject) >= 1.2


def search_wiki_batch(
    service_url: str,
    queries: list[str],
    *,
    top_k: int,
    timeout_s: float,
    max_queries_per_request: int = 512,
) -> list[dict[str, Any]]:
    endpoint = service_url.rstrip("/") + "/search_batch"
    if len(queries) > max_queries_per_request:
        rows: list[dict[str, Any]] = []
        for start in range(0, len(queries), max_queries_per_request):
            rows.extend(
                search_wiki_batch(
                    service_url,
                    queries[start : start + max_queries_per_request],
                    top_k=top_k,
                    timeout_s=timeout_s,
                    max_queries_per_request=max_queries_per_request,
                )
            )
        return rows
    with httpx.Client(timeout=timeout_s) as client:
        response = client.post(endpoint, json={"queries": queries, "top_k": top_k})
        response.raise_for_status()
        payload = response.json()
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError(f"unexpected search_batch payload keys={sorted(payload)}")
    return results


def retrieve_subject_materials(
    plan: SubjectPlan,
    seeds: SeedQueryPlan,
    *,
    service_url: str,
    top_k_per_query: int,
    timeout_s: float,
) -> list[RetrievedMaterial]:
    batch_results = search_wiki_batch(service_url, seeds.seed_queries, top_k=top_k_per_query, timeout_s=timeout_s)
    by_passage: dict[str, RetrievedMaterial] = {}
    for item in batch_results:
        query = str(item.get("query") or "")
        for hit in item.get("results", []):
            if not isinstance(hit, dict) or not should_keep_material(hit, plan.subject_id):
                continue
            passage_id = str(hit.get("passage_id") or "")
            if not passage_id:
                continue
            title = hit.get("title")
            section = hit.get("section")
            text = str(hit.get("text") or "")
            retrieval_score = float(hit.get("score") or 0.0)
            material_score = material_quality_score(text, str(title or ""), str(section or ""), plan.subject_id)
            row = RetrievedMaterial(
                source_id=stable_id("source", plan.subject_id, passage_id),
                subject=plan.subject_id,
                title=str(title) if title is not None else None,
                section=str(section) if section is not None else None,
                passage_id=passage_id,
                text=text,
                retrieval_query=query,
                retrieval_score=retrieval_score,
                material_score=material_score,
                rank=int(hit.get("rank") or 0),
            )
            previous = by_passage.get(passage_id)
            if previous is None or (row.retrieval_score + row.material_score) > (
                previous.retrieval_score + previous.material_score
            ):
                by_passage[passage_id] = row
    rows = sorted(
        by_passage.values(),
        key=lambda item: (-item.material_score, -item.retrieval_score, item.rank, item.passage_id),
    )
    return rows[: plan.max_passages]


def extraction_prompt(subject: str, material: RetrievedMaterial) -> list[dict[str, str]]:
    clipped_text = material.text[:4500]
    return [
        {
            "role": "system",
            "content": (
                "You extract reusable academic concepts from English knowledge material. "
                "This is subject-driven and corpus-driven. Use only supplied corpus material. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": f"""Subject scope: {subject}
Source title: {material.title or ""}
Source section: {material.section or ""}

Passage:
{clipped_text}

Extract 0 to 5 candidate concepts that are stable, reusable knowledge units in this subject.

Keep:
- theories, principles, laws, indicators, methods, processes, classifications, defined terms.

Reject:
- the subject name itself;
- ordinary actions;
- one-off examples;
- pure people/events/entities without a reusable academic definition;
- usage patterns;
- overly broad umbrella fields;
- composites that should be split.

For evidence_span, copy a short verbatim span from the passage when possible.

Return JSON:
{{
  "candidates": [
    {{
      "raw_name": "term as found",
      "proposed_name": "canonical-looking English name",
      "concept_type": "economic_principle|metric|model|method|process|classification|defined_term|other",
      "definition_candidate": "one sentence definition",
      "evidence_span": "short span from passage",
      "aliases": ["optional aliases"],
      "confidence": 0.0
    }}
  ]
}}""",
        },
    ]


def validate_extraction_payload(payload: dict[str, Any]) -> ExtractionPayload:
    parsed = ExtractionPayload.model_validate(payload)
    parsed.candidates = parsed.candidates[:5]
    return parsed


async def extract_candidates(
    materials: list[RetrievedMaterial],
    *,
    subject: str,
    client: SmokeModelClient,
    max_passages: int,
) -> tuple[list[RawCandidate], list[dict[str, Any]]]:
    tasks = [
        client.complete_json(
            extraction_prompt(subject, material),
            namespace=f"subject_concept.extract.{subject}.{material.source_id}",
            validate=validate_extraction_payload,
        )
        for material in materials[:max_passages]
    ]
    outputs = await asyncio.gather(*tasks)
    candidates: list[RawCandidate] = []
    extraction_logs: list[dict[str, Any]] = []
    seen_key_counts: Counter[str] = Counter()
    for material, (payload, result, error) in zip(materials[:max_passages], outputs):
        extraction_logs.append(
            {
                "source_id": material.source_id,
                "passage_id": material.passage_id,
                "title": material.title,
                "section": material.section,
                "ok": isinstance(payload, ExtractionPayload),
                "error": error,
                "latency_s": result.latency_s if result else None,
                "cached": result.cached if result else None,
                "usage": result.usage if result else {},
            }
        )
        if not isinstance(payload, ExtractionPayload):
            continue
        for draft in payload.candidates:
            proposed = draft.proposed_name.strip() or draft.raw_name.strip()
            normalized = normalize_concept_name(proposed)
            if not normalized:
                continue
            base_key = stable_id("cand", subject, material.source_id, normalized)
            seen_key_counts[base_key] += 1
            candidate_id = f"{base_key}_{seen_key_counts[base_key]}"
            candidates.append(
                RawCandidate(
                    candidate_id=candidate_id,
                    subject=subject,
                    raw_name=draft.raw_name.strip(),
                    proposed_name=proposed,
                    normalized_name=normalized,
                    concept_type=draft.concept_type.strip() or "other",
                    definition_candidate=draft.definition_candidate.strip(),
                    source_id=material.source_id,
                    passage_id=material.passage_id,
                    title=material.title,
                    section=material.section,
                    evidence_span=draft.evidence_span.strip(),
                    aliases=[alias.strip() for alias in draft.aliases if alias.strip()],
                    confidence=float(draft.confidence or 0.0),
                )
            )
    return candidates, extraction_logs


def filter_candidate(candidate: RawCandidate) -> FilteredCandidate:
    reasons: list[str] = []
    name = candidate.normalized_name
    words = name.split()
    if not name:
        reasons.append("EMPTY_NAME")
    if name in TOO_BROAD_NAMES:
        reasons.append("TOO_BROAD")
    if len(words) > 7:
        reasons.append("TOO_NARROW")
    if any(word in USAGE_WORDS for word in words):
        reasons.append("USAGE_PATTERN")
    if candidate.confidence < 0.45:
        reasons.append("LOW_CONFIDENCE")
    if len(candidate.definition_candidate) < 30:
        reasons.append("WEAK_DEFINITION")
    # Short evidence spans are common in extraction output and can be repaired
    # by the grounding stage. Do not reject only for this reason.
    if len(candidate.evidence_span) < 20 and len(candidate.definition_candidate) < 60:
        reasons.append("WEAK_EVIDENCE_SPAN")
    if name.count(" and ") >= 1 and len(words) >= 5:
        reasons.append("COMPOSITE_RISK")
    status = "IS_CONCEPT" if not reasons else reasons[0]
    return FilteredCandidate(**candidate.model_dump(mode="json"), filter_status=status, filter_reasons=reasons)


def filter_candidates(candidates: list[RawCandidate]) -> tuple[list[FilteredCandidate], list[FilteredCandidate]]:
    filtered = [filter_candidate(candidate) for candidate in candidates]
    kept = [candidate for candidate in filtered if candidate.filter_status == "IS_CONCEPT"]
    rejected = [candidate for candidate in filtered if candidate.filter_status != "IS_CONCEPT"]
    return kept, rejected


def find_evidence_span(text: str, name: str, fallback: str) -> str:
    normalized_tokens = [tok for tok in normalize_concept_name(name).split() if tok]
    if normalized_tokens:
        pattern = re.compile(r"[^.?!]*(?:" + "|".join(map(re.escape, normalized_tokens[:3])) + r")[^.?!]*[.?!]", re.I)
        match = pattern.search(text)
        if match:
            return re.sub(r"\s+", " ", match.group(0)).strip()[:700]
    return re.sub(r"\s+", " ", fallback).strip()[:700]


def name_supported_by_hit(candidate: FilteredCandidate, hit: dict[str, Any]) -> bool:
    text = normalize_concept_name(" ".join([str(hit.get("title") or ""), str(hit.get("section") or ""), str(hit.get("text") or "")]))
    tokens = [tok for tok in candidate.normalized_name.split() if len(tok) > 2]
    if not tokens:
        return False
    hits = sum(1 for tok in tokens if tok in text)
    return hits >= min(len(tokens), 2)


def ground_candidates(
    candidates: list[FilteredCandidate],
    *,
    subject: str,
    service_url: str,
    top_k: int,
    timeout_s: float,
) -> tuple[list[GroundedConcept], list[dict[str, Any]]]:
    if not candidates:
        return [], []
    label = subject_label(subject)
    queries = [f"{candidate.proposed_name} definition {label}" for candidate in candidates]
    batch_results = search_wiki_batch(service_url, queries, top_k=top_k, timeout_s=timeout_s)
    grounded: list[GroundedConcept] = []
    grounding_passages: list[dict[str, Any]] = []
    for candidate, item in zip(candidates, batch_results):
        evidence: list[GroundingEvidence] = []
        for hit in item.get("results", []):
            if not isinstance(hit, dict) or hit.get("missing"):
                continue
            grounding_passages.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "query": item.get("query"),
                    "rank": hit.get("rank"),
                    "score": hit.get("score"),
                    "passage_id": hit.get("passage_id"),
                    "title": hit.get("title"),
                    "section": hit.get("section"),
                    "text": hit.get("text"),
                }
            )
            if name_supported_by_hit(candidate, hit):
                text = str(hit.get("text") or "")
                evidence.append(
                    GroundingEvidence(
                        source_id=stable_id("source", subject, str(hit.get("passage_id") or "")),
                        passage_id=str(hit.get("passage_id") or ""),
                        title=str(hit.get("title")) if hit.get("title") is not None else None,
                        section=str(hit.get("section")) if hit.get("section") is not None else None,
                        evidence_span=find_evidence_span(text, candidate.proposed_name, candidate.evidence_span),
                        retrieval_query=str(item.get("query") or ""),
                        retrieval_score=float(hit.get("score") or 0.0),
                    )
                )
                break
        if evidence:
            status = "SUPPORTED"
            confidence = min(0.99, max(candidate.confidence, 0.72))
        elif candidate.evidence_span:
            status = "PARTIAL"
            confidence = min(0.71, candidate.confidence)
            evidence = [
                GroundingEvidence(
                    source_id=candidate.source_id,
                    passage_id=candidate.passage_id,
                    title=candidate.title,
                    section=candidate.section,
                    evidence_span=candidate.evidence_span[:700],
                    retrieval_query=None,
                    retrieval_score=None,
                )
            ]
        else:
            status = "UNSUPPORTED"
            confidence = min(0.45, candidate.confidence)
        aliases = sorted(
            {
                alias
                for alias in [*candidate.aliases, candidate.raw_name]
                if alias and normalize_concept_name(alias) != candidate.normalized_name
            },
            key=str.lower,
        )
        grounded.append(
            GroundedConcept(
                candidate_id=candidate.candidate_id,
                subject=subject,
                grounding_status=status,
                canonical_name_suggestion=candidate.proposed_name,
                normalized_name=candidate.normalized_name,
                definition=candidate.definition_candidate,
                scope=label,
                concept_type=candidate.concept_type,
                aliases=aliases,
                evidence=evidence,
                confidence=confidence,
            )
        )
    return grounded, grounding_passages


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def build_candidate_pairs(grounded: list[GroundedConcept]) -> list[CandidatePair]:
    active = [item for item in grounded if item.grounding_status in {"SUPPORTED", "PARTIAL"}]
    pairs: list[CandidatePair] = []
    by_name: dict[str, list[GroundedConcept]] = defaultdict(list)
    by_alias: dict[str, list[GroundedConcept]] = defaultdict(list)
    by_signature: dict[str, list[GroundedConcept]] = defaultdict(list)
    for item in active:
        by_name[item.normalized_name].append(item)
        signature = semantic_same_signature(item.normalized_name)
        if signature:
            by_signature[signature].append(item)
        for alias in item.aliases:
            alias_key = normalize_concept_name(alias)
            if alias_key:
                by_alias[alias_key].append(item)
    seen: set[tuple[str, str, str]] = set()

    def add_pair(a: GroundedConcept, b: GroundedConcept, reason: str) -> None:
        if a.candidate_id == b.candidate_id:
            return
        key = tuple(sorted([a.candidate_id, b.candidate_id]) + [reason])
        if key in seen:
            return
        seen.add(key)
        pairs.append(
            CandidatePair(
                candidate_a=min(a.candidate_id, b.candidate_id),
                candidate_b=max(a.candidate_id, b.candidate_id),
                relation="SAME",
                confidence=0.92 if reason == "normalized_name" else 0.86,
                blocking_reasons=[reason],
                evidence_source_ids=sorted({ev.source_id for ev in [*a.evidence, *b.evidence]})[:3],
            )
        )

    for items in by_name.values():
        for idx, a in enumerate(items):
            for b in items[idx + 1 :]:
                add_pair(a, b, "normalized_name")
    for items in by_alias.values():
        for idx, a in enumerate(items):
            for b in items[idx + 1 :]:
                add_pair(a, b, "shared_alias")
    for items in by_signature.values():
        for idx, a in enumerate(items):
            for b in items[idx + 1 :]:
                add_pair(a, b, "semantic_signature")
    return pairs


def semantic_same_signature(name: str) -> str:
    tokens = [
        tok
        for tok in normalize_concept_name(name).split()
        if tok not in {"theory", "of", "model", "concept", "principle"}
    ]
    if len(tokens) < 2:
        return ""
    return " ".join(sorted(tokens))


def pair_judge_prompt(pair: CandidatePair, by_candidate: dict[str, GroundedConcept]) -> list[dict[str, str]]:
    a = by_candidate[pair.candidate_a]
    b = by_candidate[pair.candidate_b]
    a_ev = a.evidence[0].evidence_span if a.evidence else ""
    b_ev = b.evidence[0].evidence_span if b.evidence else ""
    return [
        {
            "role": "system",
            "content": (
                "You judge canonical concept relationships inside one academic subject. "
                "Return only valid JSON. Only SAME permits merging."
            ),
        },
        {
            "role": "user",
            "content": f"""Subject: {a.subject}

Candidate A:
Name: {a.canonical_name_suggestion}
Normalized name: {a.normalized_name}
Type: {a.concept_type}
Definition: {a.definition}
Evidence: {a_ev[:800]}

Candidate B:
Name: {b.canonical_name_suggestion}
Normalized name: {b.normalized_name}
Type: {b.concept_type}
Definition: {b.definition}
Evidence: {b_ev[:800]}

Blocking reasons: {', '.join(pair.blocking_reasons)}

Judge relation:
- SAME: aliases or same concept at same granularity.
- RELATED: connected but not same.
- CONFUSABLE: often mistaken for each other; should remain separate.
- DISTINCT: no useful same/related/confusable relation.

Return JSON:
{{"relation":"SAME|RELATED|CONFUSABLE|DISTINCT","confidence":0.0,"rationale":"short reason"}}""",
        },
    ]


def validate_pair_judgement(payload: dict[str, Any]) -> PairJudgementPayload:
    parsed = PairJudgementPayload.model_validate(payload)
    relation = parsed.relation.strip().upper()
    if relation not in {"SAME", "RELATED", "CONFUSABLE", "DISTINCT"}:
        raise ValueError(f"invalid relation: {parsed.relation}")
    parsed.relation = relation
    parsed.confidence = max(0.0, min(1.0, float(parsed.confidence)))
    return parsed


async def judge_candidate_pairs_llm(
    pairs: list[CandidatePair],
    grounded: list[GroundedConcept],
    *,
    client: SmokeModelClient,
    max_pairs: int,
) -> tuple[list[CandidatePair], list[dict[str, Any]]]:
    if not pairs or max_pairs <= 0:
        return pairs, []
    by_candidate = {item.candidate_id: item for item in grounded}
    judged_pairs = pairs[:max_pairs]
    passthrough_pairs = pairs[max_pairs:]
    tasks = [
        client.complete_json(
            pair_judge_prompt(pair, by_candidate),
            namespace=f"subject_concept.pair_judge.{by_candidate[pair.candidate_a].subject}.{pair.candidate_a}.{pair.candidate_b}",
            validate=validate_pair_judgement,
        )
        for pair in judged_pairs
        if pair.candidate_a in by_candidate and pair.candidate_b in by_candidate
    ]
    outputs = await asyncio.gather(*tasks) if tasks else []
    updated: list[CandidatePair] = []
    logs: list[dict[str, Any]] = []
    for pair, (judgement, result, error) in zip(judged_pairs, outputs):
        if isinstance(judgement, PairJudgementPayload):
            updated_pair = CandidatePair(
                candidate_a=pair.candidate_a,
                candidate_b=pair.candidate_b,
                relation=judgement.relation,
                confidence=judgement.confidence,
                blocking_reasons=pair.blocking_reasons,
                evidence_source_ids=pair.evidence_source_ids,
            )
            logs.append(
                {
                    "candidate_a": pair.candidate_a,
                    "candidate_b": pair.candidate_b,
                    "relation": judgement.relation,
                    "confidence": judgement.confidence,
                    "rationale": judgement.rationale,
                    "error": None,
                    "latency_s": result.latency_s if result else None,
                    "cached": result.cached if result else None,
                    "usage": result.usage if result else {},
                }
            )
        else:
            updated_pair = pair
            logs.append(
                {
                    "candidate_a": pair.candidate_a,
                    "candidate_b": pair.candidate_b,
                    "relation": pair.relation,
                    "confidence": pair.confidence,
                    "rationale": "fallback_to_blocking_relation",
                    "error": error,
                    "latency_s": result.latency_s if result else None,
                    "cached": result.cached if result else None,
                    "usage": result.usage if result else {},
                }
            )
        updated.append(updated_pair)
    return [*updated, *passthrough_pairs], logs


def quality_scores(row: ConceptRegistryRow, evidence_count: int, cluster_size: int) -> ConceptQualityRow:
    word_count = len(row.normalized_name.split())
    evidence_support = 5 if row.status == "ACTIVE" and evidence_count else 3 if evidence_count else 1
    granularity = 5 if 1 <= word_count <= 5 and row.normalized_name not in TOO_BROAD_NAMES else 3
    subject_relevance = 5 if evidence_count else 3
    canonicalization = 5 if row.canonical_name == row.canonical_name.lower() else 4
    duplicate_risk = 1 if cluster_size > 1 else 2
    status = "ACTIVE" if min(evidence_support, granularity, subject_relevance, canonicalization) >= 4 else "provisional"
    return ConceptQualityRow(
        concept_id=row.concept_id,
        evidence_support=evidence_support,
        granularity=granularity,
        subject_relevance=subject_relevance,
        canonicalization=canonicalization,
        duplicate_risk=duplicate_risk,
        status=status,
    )


def build_registry(
    grounded: list[GroundedConcept],
    pairs: list[CandidatePair],
    *,
    subject: str,
    target_active: int,
) -> tuple[
    list[ConceptRegistryRow],
    list[ConceptEvidenceRow],
    list[MergeRedirectRow],
    list[ConceptRelationRow],
    list[ConceptQualityRow],
    list[ConceptIndexRow],
]:
    active = [item for item in grounded if item.grounding_status in {"SUPPORTED", "PARTIAL"}]
    uf = UnionFind([item.candidate_id for item in active])
    for pair in pairs:
        if pair.relation == "SAME" and pair.confidence >= 0.85:
            uf.union(pair.candidate_a, pair.candidate_b)
    by_id = {item.candidate_id: item for item in active}
    clusters: dict[str, list[GroundedConcept]] = defaultdict(list)
    for item in active:
        clusters[uf.find(item.candidate_id)].append(item)

    registry: list[ConceptRegistryRow] = []
    evidence_rows: list[ConceptEvidenceRow] = []
    redirects: list[MergeRedirectRow] = []
    quality_rows: list[ConceptQualityRow] = []
    prefix = subject_prefix(subject)
    used_ids: Counter[str] = Counter()

    sorted_clusters = sorted(
        clusters.values(),
        key=lambda items: (
            -max(item.confidence for item in items),
            -sum(1 for item in items if item.grounding_status == "SUPPORTED"),
            min(item.normalized_name for item in items),
        ),
    )
    for items in sorted_clusters:
        if len([row for row in registry if row.status == "ACTIVE"]) >= target_active:
            break
        best = sorted(items, key=lambda item: (-item.confidence, item.normalized_name))[0]
        base_id = f"{prefix}-{slug_name(best.normalized_name)}"
        used_ids[base_id] += 1
        concept_id = base_id if used_ids[base_id] == 1 else f"{base_id}-{used_ids[base_id]}"
        aliases = sorted(
            {
                alias
                for item in items
                for alias in [item.canonical_name_suggestion, *item.aliases]
                if normalize_concept_name(alias) != best.normalized_name
            },
            key=str.lower,
        )
        evidence = [ev for item in items for ev in item.evidence]
        status = "ACTIVE" if best.grounding_status == "SUPPORTED" and evidence else "provisional"
        row = ConceptRegistryRow(
            concept_id=concept_id,
            subject_scopes=[subject],
            canonical_name=best.normalized_name,
            normalized_name=best.normalized_name,
            aliases=aliases[:8],
            definition=best.definition,
            concept_type=best.concept_type,
            status=status,
        )
        quality = quality_scores(row, len(evidence), len(items))
        if quality.status != row.status and quality.status == "provisional":
            row.status = "provisional"
        registry.append(row)
        quality_rows.append(quality_scores(row, len(evidence), len(items)))
        for item in items:
            redirects.append(MergeRedirectRow(from_candidate_id=item.candidate_id, to_concept_id=concept_id, confidence=item.confidence))
        for ev in evidence[:3]:
            evidence_rows.append(
                ConceptEvidenceRow(
                    concept_id=concept_id,
                    source_id=ev.source_id,
                    passage_id=ev.passage_id,
                    title=ev.title,
                    section=ev.section,
                    evidence_span=ev.evidence_span,
                )
            )

    concept_by_candidate = {redirect.from_candidate_id: redirect.to_concept_id for redirect in redirects}
    relations: list[ConceptRelationRow] = []
    seen_relation: set[tuple[str, str]] = set()
    evidence_by_concept: dict[str, set[str]] = defaultdict(set)
    for row in evidence_rows:
        evidence_by_concept[row.concept_id].add(row.source_id)
    for source_id, concept_ids in _concepts_by_shared_evidence(evidence_rows).items():
        sorted_ids = sorted(concept_ids)
        for idx, source_concept_id in enumerate(sorted_ids):
            for target_concept_id in sorted_ids[idx + 1 :]:
                key = (source_concept_id, target_concept_id)
                if key in seen_relation:
                    continue
                seen_relation.add(key)
                relations.append(
                    ConceptRelationRow(
                        source_concept_id=source_concept_id,
                        relation="RELATED_TO",
                        target_concept_id=target_concept_id,
                        evidence_source_ids=[source_id],
                        confidence=0.55,
                    )
                )
                if len(relations) >= 100:
                    break
            if len(relations) >= 100:
                break
        if len(relations) >= 100:
            break

    index_rows = [
        ConceptIndexRow(
            concept_id=row.concept_id,
            index_text=build_index_text(row, relations),
        )
        for row in registry
    ]
    return registry, evidence_rows, redirects, relations, quality_rows, index_rows


def _concepts_by_shared_evidence(evidence_rows: list[ConceptEvidenceRow]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in evidence_rows:
        grouped[row.source_id].add(row.concept_id)
    return {source_id: ids for source_id, ids in grouped.items() if len(ids) > 1}


def build_index_text(row: ConceptRegistryRow, relations: list[ConceptRelationRow]) -> str:
    related: list[str] = []
    relation_ids = {
        rel.target_concept_id
        for rel in relations
        if rel.source_concept_id == row.concept_id and rel.relation in {"RELATED_TO", "CONFUSABLE_WITH"}
    }
    if relation_ids:
        related = sorted(relation_ids)[:8]
    parts = [
        f"Concept: {row.canonical_name}",
        f"Aliases: {', '.join(row.aliases) if row.aliases else 'None'}",
        f"Definition: {row.definition}",
        f"Subject: {', '.join(row.subject_scopes)}",
        f"Concept type: {row.concept_type}",
    ]
    if related:
        parts.append(f"Related concept ids: {', '.join(related)}")
    return "\n".join(parts)


def concept_lookup(registry: list[ConceptRegistryRow]) -> dict[str, ConceptRegistryRow]:
    return {row.concept_id: row for row in registry}


def build_usage_jobs(
    registry: list[ConceptRegistryRow],
    relations: list[ConceptRelationRow],
    *,
    subject: str,
    max_concepts: int,
    max_jobs: int,
) -> list[UsageJobRow]:
    by_concept = concept_lookup(registry)
    related_names: dict[str, tuple[str, str]] = {}
    for relation in relations:
        if relation.source_concept_id in by_concept and relation.target_concept_id in by_concept:
            related_names.setdefault(
                relation.source_concept_id,
                (relation.target_concept_id, by_concept[relation.target_concept_id].canonical_name),
            )
            related_names.setdefault(
                relation.target_concept_id,
                (relation.source_concept_id, by_concept[relation.source_concept_id].canonical_name),
            )

    jobs: list[UsageJobRow] = []
    for concept in [row for row in registry if row.status == "ACTIVE"][:max_concepts]:
        job_specs = [
            (
                "application_rules",
                f"{concept.canonical_name} application conditions procedure {subject_label(subject)}",
                None,
                None,
            ),
            (
                "boundary_exceptions",
                f"{concept.canonical_name} scope exception boundary when not applicable {subject_label(subject)}",
                None,
                None,
            ),
        ]
        related = related_names.get(concept.concept_id)
        if related:
            other_id, other_name = related
            query = f"{concept.canonical_name} vs {other_name} difference common misconception {subject_label(subject)}"
            job_specs.append(("confusable_distinction", query, other_id, other_name))
        else:
            query = f"{concept.canonical_name} common misconception difference verification rule {subject_label(subject)}"
            job_specs.append(("confusable_distinction", query, None, None))

        for job_type, query, other_id, other_name in job_specs:
            jobs.append(
                UsageJobRow(
                    usage_job_id=stable_id("usage_job", subject, concept.concept_id, job_type, query),
                    subject=subject,
                    concept_id=concept.concept_id,
                    concept=concept.canonical_name,
                    job_type=job_type,
                    retrieval_query=query,
                    confusable_concept_id=other_id,
                    confusable_concept=other_name,
                )
            )
            if len(jobs) >= max_jobs:
                return jobs
    return jobs


def usage_material_scores(job: UsageJobRow, hit: dict[str, Any]) -> tuple[int, int, bool, bool, bool, str]:
    text = str(hit.get("text") or "")
    lowered = text.lower()
    concept_terms = [tok for tok in normalize_concept_name(job.concept).split() if len(tok) > 2]
    query_terms = [tok for tok in normalize_concept_name(job.retrieval_query).split() if len(tok) > 3]
    concept_hits = sum(1 for term in concept_terms if term in lowered)
    usage_hits = sum(1 for term in query_terms if term in lowered)
    concept_relevance = max(1, min(5, 1 + concept_hits * 2))
    usage_relevance = max(1, min(5, 1 + usage_hits // 2))
    supports_rule = any(term in lowered for term in ["condition", "if ", "when ", "therefore", "because", "rule", "principle", "applies"])
    supports_boundary = any(term in lowered for term in ["exception", "not ", "however", "although", "limit", "boundary", "constraint", "whereas"])
    generic_definition_only = concept_relevance >= 4 and usage_relevance <= 2 and not (supports_rule or supports_boundary)
    decision = "KEEP" if concept_relevance >= 3 and usage_relevance >= 2 and not generic_definition_only else "REJECT"
    return concept_relevance, usage_relevance, supports_rule, supports_boundary, generic_definition_only, decision


def retrieve_usage_materials(
    jobs: list[UsageJobRow],
    *,
    service_url: str,
    top_k: int,
    final_per_job: int,
    timeout_s: float,
) -> tuple[list[UsageMaterialRow], list[dict[str, Any]]]:
    if not jobs:
        return [], []
    batch_results = search_wiki_batch(service_url, [job.retrieval_query for job in jobs], top_k=top_k, timeout_s=timeout_s)
    materials: list[UsageMaterialRow] = []
    critic_rows: list[dict[str, Any]] = []
    for job, item in zip(jobs, batch_results):
        kept_for_job: list[UsageMaterialRow] = []
        for hit in item.get("results", []):
            if not isinstance(hit, dict) or hit.get("missing"):
                continue
            text = str(hit.get("text") or "").strip()
            if len(text) < 200:
                continue
            concept_rel, usage_rel, supports_rule, supports_boundary, generic_only, decision = usage_material_scores(job, hit)
            critic = {
                "usage_job_id": job.usage_job_id,
                "source_id": stable_id("usage_source", job.usage_job_id, str(hit.get("passage_id") or "")),
                "passage_id": str(hit.get("passage_id") or ""),
                "title": hit.get("title"),
                "section": hit.get("section"),
                "concept_relevance": concept_rel,
                "usage_relevance": usage_rel,
                "supports_rule": supports_rule,
                "supports_boundary": supports_boundary,
                "generic_definition_only": generic_only,
                "decision": decision,
            }
            critic_rows.append(critic)
            if decision != "KEEP":
                continue
            kept_for_job.append(
                UsageMaterialRow(
                    usage_job_id=job.usage_job_id,
                    source_id=critic["source_id"],
                    subject=job.subject,
                    concept_id=job.concept_id,
                    concept=job.concept,
                    job_type=job.job_type,
                    title=str(hit.get("title")) if hit.get("title") is not None else None,
                    section=str(hit.get("section")) if hit.get("section") is not None else None,
                    passage_id=str(hit.get("passage_id") or ""),
                    text=text,
                    retrieval_query=job.retrieval_query,
                    retrieval_score=float(hit.get("score") or 0.0),
                    concept_relevance=concept_rel,
                    usage_relevance=usage_rel,
                    supports_rule=supports_rule,
                    supports_boundary=supports_boundary,
                    generic_definition_only=generic_only,
                    decision=decision,
                    rank=int(hit.get("rank") or 0),
                )
            )
        kept_for_job = sorted(
            kept_for_job,
            key=lambda row: (-(row.concept_relevance + row.usage_relevance), -row.retrieval_score, row.rank, row.passage_id),
        )[:final_per_job]
        materials.extend(kept_for_job)
    return materials, critic_rows


def usage_card_prompt(
    job: UsageJobRow,
    concept: ConceptRegistryRow,
    materials: list[UsageMaterialRow],
) -> list[dict[str, str]]:
    material_block = "\n\n".join(
        [
            f"Source ID: {material.source_id}\n"
            f"Title: {material.title or ''}\n"
            f"Section: {material.section or ''}\n"
            f"Text: {material.text[:1600]}"
            for material in materials
        ]
    )
    confusable = job.confusable_concept or "common misconception / adjacent concept"
    return [
        {
            "role": "system",
            "content": (
                "You build source-grounded procedural Usage Cards for a corpus-derived bank. "
                "Use only the supplied English materials. Do not mention external assessment data. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": f"""Subject: {job.subject}
Concept ID: {job.concept_id}
Concept: {job.concept}
Definition: {concept.definition}
Usage job type: {job.job_type}
Usage retrieval query: {job.retrieval_query}
Confusable / adjacent concept: {confusable}

Materials:
{material_block}

Create 1 procedural Usage Card candidate.

Requirements:
- The card must be executable: concrete trigger conditions and decision steps.
- Do not restate only the definition.
- Do not invent rules unsupported by the supplied materials.
- Every important trigger, procedure, boundary, or verification rule should have an evidence_claim with source_ids from the materials.
- If materials are insufficient, return an empty cards list.

Return JSON:
{{
  "cards": [
    {{
      "usage_pattern": "short pattern name",
      "usage_signature": "compact routing signature",
      "concept_boundary": "when this concept/card applies",
      "trigger_conditions": ["..."],
      "decision_procedure": ["..."],
      "failure_boundaries": ["..."],
      "verification_rules": ["..."],
      "evidence_claims": [
        {{
          "field": "concept_boundary|trigger_conditions|decision_procedure|failure_boundaries|verification_rules",
          "claim": "atomic claim",
          "source_ids": ["source id from materials"],
          "support_type": "direct|derived"
        }}
      ]
    }}
  ]
}}""",
        },
    ]


def validate_usage_extraction_payload(payload: dict[str, Any]) -> UsageExtractionPayload:
    parsed = UsageExtractionPayload.model_validate(payload)
    parsed.cards = parsed.cards[:3]
    return parsed


async def extract_usage_cards(
    jobs: list[UsageJobRow],
    registry: list[ConceptRegistryRow],
    materials: list[UsageMaterialRow],
    *,
    client: SmokeModelClient,
) -> tuple[list[UsageCardRow], list[dict[str, Any]]]:
    by_concept = concept_lookup(registry)
    materials_by_job: dict[str, list[UsageMaterialRow]] = defaultdict(list)
    for material in materials:
        materials_by_job[material.usage_job_id].append(material)
    runnable_jobs = [job for job in jobs if materials_by_job.get(job.usage_job_id) and job.concept_id in by_concept]
    tasks = [
        client.complete_json(
            usage_card_prompt(job, by_concept[job.concept_id], materials_by_job[job.usage_job_id]),
            namespace=f"subject_usage_card.extract.{job.subject}.{job.usage_job_id}",
            validate=validate_usage_extraction_payload,
        )
        for job in runnable_jobs
    ]
    outputs = await asyncio.gather(*tasks) if tasks else []
    cards: list[UsageCardRow] = []
    logs: list[dict[str, Any]] = []
    for job, (payload, result, error) in zip(runnable_jobs, outputs):
        valid_sources = {material.source_id for material in materials_by_job[job.usage_job_id]}
        logs.append(
            {
                "usage_job_id": job.usage_job_id,
                "concept_id": job.concept_id,
                "job_type": job.job_type,
                "ok": isinstance(payload, UsageExtractionPayload),
                "error": error,
                "latency_s": result.latency_s if result else None,
                "cached": result.cached if result else None,
                "usage": result.usage if result else {},
            }
        )
        if not isinstance(payload, UsageExtractionPayload):
            continue
        for idx, draft in enumerate(payload.cards, start=1):
            source_ids = sorted(
                {
                    source_id
                    for claim in draft.evidence_claims
                    for source_id in claim.source_ids
                    if source_id in valid_sources
                }
            )
            usage_id = stable_id("usage", job.subject, job.concept_id, job.job_type, draft.usage_pattern, idx)
            normalized_claims: list[EvidenceClaimDraft] = []
            for claim in draft.evidence_claims:
                normalized_claims.append(
                    EvidenceClaimDraft(
                        field=claim.field,
                        claim=claim.claim,
                        source_ids=[source_id for source_id in claim.source_ids if source_id in valid_sources],
                        support_type=claim.support_type if claim.support_type in {"direct", "derived"} else "direct",
                    )
                )
            cards.append(
                UsageCardRow(
                    usage_id=usage_id,
                    usage_job_id=job.usage_job_id,
                    subject=job.subject,
                    concept_id=job.concept_id,
                    concept=job.concept,
                    usage_pattern=draft.usage_pattern.strip(),
                    usage_signature=draft.usage_signature.strip(),
                    concept_boundary=draft.concept_boundary.strip(),
                    trigger_conditions=[item.strip() for item in draft.trigger_conditions if item.strip()][:6],
                    decision_procedure=[item.strip() for item in draft.decision_procedure if item.strip()][:8],
                    failure_boundaries=[item.strip() for item in draft.failure_boundaries if item.strip()][:8],
                    verification_rules=[item.strip() for item in draft.verification_rules if item.strip()][:8],
                    evidence_claims=normalized_claims,
                    source_ids=source_ids,
                    status="provisional",
                )
            )
    return cards, logs


def verify_usage_cards(cards: list[UsageCardRow], materials: list[UsageMaterialRow]) -> tuple[list[UsageCardRow], list[UsageCardClaimRow], list[RejectedItemRow]]:
    material_source_ids = {material.source_id for material in materials}
    verified_cards: list[UsageCardRow] = []
    claim_rows: list[UsageCardClaimRow] = []
    rejected: list[RejectedItemRow] = []
    for card in cards:
        accepted_core_claims = 0
        rejected_core_claims = 0
        has_boundary_support = False
        for idx, claim in enumerate(card.evidence_claims, start=1):
            source_ids = [source_id for source_id in claim.source_ids if source_id in material_source_ids]
            is_core = claim.field in {"concept_boundary", "decision_procedure", "trigger_conditions", "verification_rules"}
            if source_ids and len(claim.claim.strip()) >= 20:
                decision = "ACCEPT_DERIVED" if claim.support_type == "derived" else "ACCEPT"
                if is_core:
                    accepted_core_claims += 1
                if claim.field == "concept_boundary":
                    has_boundary_support = True
            else:
                decision = "REJECT"
                if is_core:
                    rejected_core_claims += 1
            claim_rows.append(
                UsageCardClaimRow(
                    claim_id=stable_id("claim", card.usage_id, idx, claim.field, claim.claim),
                    usage_id=card.usage_id,
                    usage_job_id=card.usage_job_id,
                    concept_id=card.concept_id,
                    field=claim.field,
                    claim=claim.claim,
                    supporting_source_ids=source_ids,
                    support_type=claim.support_type,
                    contradiction=False,
                    decision=decision,
                )
            )
        too_thin = not card.trigger_conditions or not card.decision_procedure or not card.concept_boundary
        if has_boundary_support and accepted_core_claims > 0 and rejected_core_claims == 0 and not too_thin:
            card.status = "active"
            verified_cards.append(card)
        else:
            card.status = "rejected"
            reason = "unsupported_core_claim" if rejected_core_claims else "missing_boundary_or_procedure_support" if not has_boundary_support else "too_thin"
            rejected.append(
                RejectedItemRow(
                    item_id=card.usage_id,
                    item_type="usage_card",
                    stage="claim_verification",
                    reason=reason,
                    payload=card.model_dump(mode="json"),
                )
            )
    return verified_cards, claim_rows, rejected


def consolidate_usage_cards(cards: list[UsageCardRow]) -> tuple[list[UsageCardRow], list[ConsolidationRow], list[RejectedItemRow]]:
    active: list[UsageCardRow] = []
    consolidations: list[ConsolidationRow] = []
    rejected: list[RejectedItemRow] = []
    seen: dict[tuple[str, str], UsageCardRow] = {}
    for card in cards:
        key = (card.concept_id, normalize_concept_name(card.usage_pattern or card.usage_signature))
        existing = seen.get(key)
        if existing is None:
            seen[key] = card
            active.append(card)
            consolidations.append(
                ConsolidationRow(
                    usage_id=card.usage_id,
                    subject=card.subject,
                    concept_id=card.concept_id,
                    before=None,
                    candidate=card.model_dump(mode="json"),
                    decision="CREATE",
                    patch={},
                    after=card.model_dump(mode="json"),
                    source_provenance=card.source_ids,
                )
            )
            continue
        if set(card.source_ids) - set(existing.source_ids):
            patch = {"source_ids": {"add": sorted(set(card.source_ids) - set(existing.source_ids))}}
            existing.source_ids = sorted(set(existing.source_ids) | set(card.source_ids))
            consolidations.append(
                ConsolidationRow(
                    usage_id=card.usage_id,
                    subject=card.subject,
                    concept_id=card.concept_id,
                    before=existing.model_dump(mode="json"),
                    candidate=card.model_dump(mode="json"),
                    decision="MINIMAL_UPDATE",
                    patch=patch,
                    after=existing.model_dump(mode="json"),
                    source_provenance=card.source_ids,
                )
            )
        else:
            consolidations.append(
                ConsolidationRow(
                    usage_id=card.usage_id,
                    subject=card.subject,
                    concept_id=card.concept_id,
                    before=existing.model_dump(mode="json"),
                    candidate=card.model_dump(mode="json"),
                    decision="NO_OP",
                    patch={},
                    after=existing.model_dump(mode="json"),
                    source_provenance=card.source_ids,
                )
            )
            rejected.append(
                RejectedItemRow(
                    item_id=card.usage_id,
                    item_type="usage_card",
                    stage="minimal_consolidation",
                    reason="duplicate_no_op",
                    payload=card.model_dump(mode="json"),
                )
            )
    return active, consolidations, rejected


def build_usage_index_rows(cards: list[UsageCardRow]) -> list[UsageIndexRow]:
    rows: list[UsageIndexRow] = []
    for card in cards:
        index_key = "\n".join(
            [
                f"Subject: {card.subject}",
                f"Concept: {card.concept}",
                f"Usage Pattern: {card.usage_pattern}",
                f"Usage Signature: {card.usage_signature}",
                f"Concept Boundary: {card.concept_boundary}",
                "Trigger Conditions:",
                *[f"- {item}" for item in card.trigger_conditions],
            ]
        )
        payload = "\n".join(
            [
                "Decision Procedure:",
                *[f"- {item}" for item in card.decision_procedure],
                "Failure Boundaries:",
                *[f"- {item}" for item in card.failure_boundaries],
                "Verification Rules:",
                *[f"- {item}" for item in card.verification_rules],
            ]
        )
        rows.append(
            UsageIndexRow(
                usage_id=card.usage_id,
                subject=card.subject,
                concept_id=card.concept_id,
                index_key=index_key,
                payload=payload,
                status=card.status,
            )
        )
    return rows


def embed_usage_index_keys(
    rows: list[UsageIndexRow],
    *,
    service_url: str,
    timeout_s: float,
) -> tuple[list[list[float]], int]:
    endpoint = service_url.rstrip("/") + "/embed_query"
    vectors: list[list[float]] = []
    dimension: int | None = None
    with httpx.Client(timeout=timeout_s) as client:
        for row in rows:
            response = client.post(endpoint, json={"query": row.index_key})
            response.raise_for_status()
            payload = response.json()
            vector = payload.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise RuntimeError(f"embed_query returned no embedding for usage_id={row.usage_id}")
            values = [float(item) for item in vector]
            if dimension is None:
                dimension = int(payload.get("dimension") or len(values))
            if len(values) != dimension:
                raise RuntimeError(f"embedding dim mismatch for usage_id={row.usage_id}: {len(values)} != {dimension}")
            vectors.append(values)
    return vectors, int(dimension or 0)


def write_usage_faiss_index(
    rows: list[UsageIndexRow],
    *,
    usage_index_dir: Path,
    service_url: str,
    timeout_s: float,
) -> UsageIndexVectorMeta:
    import faiss
    import numpy as np

    usage_index_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        dimension = 0
        index = faiss.IndexFlatIP(1)
        faiss_path = usage_index_dir / "usage_index.faiss"
        faiss.write_index(index, str(faiss_path))
        ids_path = usage_index_dir / "usage_index_ids.jsonl"
        write_jsonl(ids_path, [])
        return UsageIndexVectorMeta(
            subject=usage_index_dir.name,
            index_type="faiss.IndexFlatIP",
            metric="inner_product_on_normalized_embeddings",
            dimension=dimension,
            count=0,
            embedding_endpoint=service_url.rstrip("/") + "/embed_query",
            embedding_input="UsageIndexRow.index_key",
            faiss_path=str(faiss_path),
            ids_path=str(ids_path),
        )

    vectors, dimension = embed_usage_index_keys(rows, service_url=service_url, timeout_s=timeout_s)
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    array = array / np.maximum(norms, 1e-12)
    index = faiss.IndexFlatIP(dimension)
    index.add(array)
    faiss_path = usage_index_dir / "usage_index.faiss"
    faiss.write_index(index, str(faiss_path))
    ids_path = usage_index_dir / "usage_index_ids.jsonl"
    write_jsonl(
        ids_path,
        [
            {
                "row_id": idx,
                "usage_id": row.usage_id,
                "concept_id": row.concept_id,
                "subject": row.subject,
                "status": row.status,
            }
            for idx, row in enumerate(rows)
        ],
    )
    meta = UsageIndexVectorMeta(
        subject=rows[0].subject,
        index_type="faiss.IndexFlatIP",
        metric="inner_product_on_normalized_embeddings",
        dimension=dimension,
        count=len(rows),
        embedding_endpoint=service_url.rstrip("/") + "/embed_query",
        embedding_input="UsageIndexRow.index_key",
        faiss_path=str(faiss_path),
        ids_path=str(ids_path),
    )
    (usage_index_dir / "usage_index_meta.json").write_text(
        json.dumps(meta.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return meta


def write_evidence_sections_parquet(out: Path, concept_materials: list[RetrievedMaterial], usage_materials: list[UsageMaterialRow]) -> None:
    rows: dict[str, dict[str, Any]] = {}
    for material in concept_materials:
        rows[material.source_id] = {
            "source_id": material.source_id,
            "source_kind": "concept_discovery",
            "subject": material.subject,
            "passage_id": material.passage_id,
            "title": material.title,
            "section": material.section,
            "text": material.text,
            "retrieval_query": material.retrieval_query,
            "retrieval_score": material.retrieval_score,
        }
    for material in usage_materials:
        rows[material.source_id] = {
            "source_id": material.source_id,
            "source_kind": "usage_material",
            "subject": material.subject,
            "passage_id": material.passage_id,
            "title": material.title,
            "section": material.section,
            "text": material.text,
            "retrieval_query": material.retrieval_query,
            "retrieval_score": material.retrieval_score,
            "usage_job_id": material.usage_job_id,
            "concept_id": material.concept_id,
        }
    import pandas as pd

    path = out / "evidence_sections.parquet"
    pd.DataFrame(list(rows.values())).to_parquet(path, index=False)


def build_bank_manifest(
    *,
    args: argparse.Namespace,
    plan: SubjectPlan,
    registry: list[ConceptRegistryRow],
    usage_cards: list[UsageCardRow],
    rejected_items: list[RejectedItemRow],
    evidence_section_count: int,
    usage_index_meta: UsageIndexVectorMeta | None,
) -> dict[str, Any]:
    return {
        "bank_version": args.bank_version,
        "source_snapshot": "wikipedia-en-2026-07-01-sherlock-faiss",
        "subjects": 1,
        "subject_ids": [plan.subject_id],
        "concept_count": len(registry),
        "active_concept_count": sum(1 for row in registry if row.status == "ACTIVE"),
        "active_card_count": sum(1 for row in usage_cards if row.status == "active"),
        "provisional_card_count": sum(1 for row in usage_cards if row.status == "provisional"),
        "rejected_card_count": sum(1 for row in rejected_items if row.item_type == "usage_card"),
        "evidence_section_count": evidence_section_count,
        "embedding_model": "Qwen3-Embedding-4B for local Wikipag retrieval",
        "usage_index": usage_index_meta.model_dump(mode="json") if usage_index_meta else None,
        "construction_model": args.model,
        "top_k": args.index_top_k,
        "similarity_gate": False,
        "construction_cutoff": datetime.now(timezone.utc).isoformat(),
        "source_corpus_only": True,
    }


def render_report(
    summary: dict[str, Any],
    registry: list[ConceptRegistryRow],
    rejected: list[FilteredCandidate],
    usage_cards: list[UsageCardRow] | None = None,
) -> str:
    lines = [
        "# Subject-driven Usage Bank Construction Smoke",
        "",
        "This smoke uses the subject source only for subject scope and budget. Concepts and Usage Cards are built from English Wikipedia passages.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Active/provisional concepts",
        "",
        "| concept_id | status | name | type |",
        "|---|---|---|---|",
    ]
    for row in registry:
        lines.append(f"| `{row.concept_id}` | {row.status} | {row.canonical_name} | {row.concept_type} |")
    if usage_cards is not None:
        lines.extend(["", "## Active usage cards", "", "| usage_id | concept | pattern | status |", "|---|---|---|---|"])
        for card in usage_cards:
            lines.append(f"| `{card.usage_id}` | {card.concept} | {card.usage_pattern} | {card.status} |")
    lines.extend(["", "## Top rejected candidates", "", "| candidate | status | reasons |", "|---|---|---|"])
    for item in rejected[:20]:
        lines.append(f"| {item.proposed_name} | {item.filter_status} | {', '.join(item.filter_reasons)} |")
    return "\n".join(lines) + "\n"


async def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = build_subject_plan(args)
    seeds = generate_seed_queries(plan.subject_id, limit=args.max_seed_queries)
    write_jsonl(out / "subject_plan.jsonl", [plan])
    write_jsonl(out / "seed_queries.jsonl", [seeds])

    materials = retrieve_subject_materials(
        plan,
        seeds,
        service_url=args.wikipag_service_url,
        top_k_per_query=args.top_k_per_query,
        timeout_s=args.retrieval_timeout_s,
    )
    write_jsonl(out / "retrieved_passages.jsonl", materials)

    model_config = ModelConfig(
        name=args.model,
        max_completion_tokens=args.max_completion_tokens,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
    )
    client = SmokeModelClient.from_config(
        model_config,
        cache_dir=out / "cache" / "model",
        raw_dir=out / "logs" / "model_raw",
        timeout_s=args.model_timeout_s,
    )
    try:
        raw_candidates, extraction_logs = await extract_candidates(
            materials,
            subject=plan.subject_id,
            client=client,
            max_passages=args.max_extraction_passages,
        )
        write_jsonl(out / "extraction_logs.jsonl", extraction_logs)
        write_jsonl(out / "candidate_concepts.raw.jsonl", raw_candidates)

        kept, rejected = filter_candidates(raw_candidates)
        kept = sorted(kept, key=lambda item: (-item.confidence, item.normalized_name))[: args.max_grounding_candidates]
        write_jsonl(out / "candidate_concepts.filtered.jsonl", kept)
        write_jsonl(out / "candidate_concepts.rejected.jsonl", rejected)

        grounded, grounding_passages = ground_candidates(
            kept,
            subject=plan.subject_id,
            service_url=args.wikipag_service_url,
            top_k=args.grounding_top_k,
            timeout_s=args.retrieval_timeout_s,
        )
        write_jsonl(out / "grounded_concepts.jsonl", grounded)
        write_jsonl(out / "grounding_passages.jsonl", grounding_passages)

        pairs = build_candidate_pairs(grounded)
        pair_judge_logs: list[dict[str, Any]] = []
        if not args.skip_llm_pair_judge:
            pairs, pair_judge_logs = await judge_candidate_pairs_llm(
                pairs,
                grounded,
                client=client,
                max_pairs=args.max_pair_judge_pairs,
            )
        write_jsonl(out / "candidate_pairs.jsonl", pairs)
        write_jsonl(out / "candidate_pair_judgements.jsonl", pair_judge_logs)
        registry, evidence, redirects, relations, quality, index_rows = build_registry(
            grounded,
            pairs,
            subject=plan.subject_id,
            target_active=plan.target_active_concepts,
        )
        write_jsonl(out / "concept_registry.jsonl", registry)
        write_jsonl(out / "concept_evidence.jsonl", evidence)
        write_jsonl(out / "merge_redirects.jsonl", redirects)
        write_jsonl(out / "concept_relations.jsonl", relations)
        write_jsonl(out / "concept_quality.jsonl", quality)
        write_jsonl(out / "concept_index.jsonl", index_rows)

        usage_jobs = build_usage_jobs(
            registry,
            relations,
            subject=plan.subject_id,
            max_concepts=args.max_usage_concepts,
            max_jobs=args.max_usage_jobs,
        )
        write_jsonl(out / "usage_jobs.jsonl", usage_jobs)
        usage_materials, usage_material_critic = retrieve_usage_materials(
            usage_jobs,
            service_url=args.wikipag_service_url,
            top_k=args.usage_retrieval_top_k,
            final_per_job=args.final_materials_per_usage_job,
            timeout_s=args.retrieval_timeout_s,
        )
        write_jsonl(out / "usage_materials.jsonl", usage_materials)
        write_jsonl(out / "usage_material_critic.jsonl", usage_material_critic)

        usage_card_candidates, usage_extraction_logs = await extract_usage_cards(
            usage_jobs,
            registry,
            usage_materials,
            client=client,
        )
    finally:
        await client.aclose()

    write_jsonl(out / "usage_card_extraction_logs.jsonl", usage_extraction_logs)
    write_jsonl(out / "usage_cards.raw.jsonl", usage_card_candidates)
    verified_cards, usage_claims, verification_rejections = verify_usage_cards(usage_card_candidates, usage_materials)
    consolidated_cards, consolidation_rows, consolidation_rejections = consolidate_usage_cards(verified_cards)
    rejected_items: list[RejectedItemRow] = [
        *[
            RejectedItemRow(
                item_id=item.candidate_id,
                item_type="concept_candidate",
                stage="concept_filter",
                reason=item.filter_status,
                payload=item.model_dump(mode="json"),
            )
            for item in rejected
        ],
        *verification_rejections,
        *consolidation_rejections,
    ]
    write_jsonl(out / "usage_cards.jsonl", consolidated_cards)
    write_jsonl(out / "usage_card_claims.jsonl", usage_claims)
    write_jsonl(out / "usage_consolidation.jsonl", consolidation_rows)
    write_jsonl(out / "rejected_items.jsonl", rejected_items)

    usage_index_rows = build_usage_index_rows(consolidated_cards)
    usage_index_dir = out / "usage_indexes" / plan.subject_id
    write_jsonl(usage_index_dir / "usage_index.jsonl", usage_index_rows)
    write_jsonl(out / "usage_index.jsonl", usage_index_rows)
    usage_index_meta = None
    if not args.skip_faiss_usage_index:
        usage_index_meta = write_usage_faiss_index(
            usage_index_rows,
            usage_index_dir=usage_index_dir,
            service_url=args.wikipag_service_url,
            timeout_s=args.embedding_timeout_s,
        )
    write_evidence_sections_parquet(out, materials, usage_materials)
    evidence_section_count = len({material.source_id for material in materials} | {material.source_id for material in usage_materials})

    build_events = [
        BuildEventRow(
            event_id=stable_id("event", plan.subject_id, stage),
            stage=stage,
            item_id=plan.subject_id,
            status="ok",
            message=message,
            payload=payload,
        )
        for stage, message, payload in [
            ("subject_planning", "subject plan written", {"subject": plan.subject_id}),
            ("concept_discovery", "concept candidates extracted", {"raw_candidate_count": len(raw_candidates)}),
            ("concept_pair_judgement", "candidate concept pairs judged", {"candidate_pair_judgement_count": len(pair_judge_logs)}),
            ("concept_canonicalization", "canonical concepts written", {"concept_count": len(registry)}),
            ("usage_material_retrieval", "usage materials retrieved", {"usage_material_count": len(usage_materials)}),
            ("usage_card_extraction", "usage card candidates extracted", {"usage_card_candidate_count": len(usage_card_candidates)}),
            ("claim_verification", "usage card claims verified", {"claim_count": len(usage_claims)}),
            ("minimal_consolidation", "usage cards consolidated", {"active_card_count": len(consolidated_cards)}),
            ("usage_index", "usage index text written", {"usage_index_count": len(usage_index_rows)}),
            (
                "usage_faiss_index",
                "usage FAISS vector index written" if usage_index_meta else "usage FAISS vector index skipped",
                usage_index_meta.model_dump(mode="json") if usage_index_meta else {"skipped": True},
            ),
        ]
    ]
    write_jsonl(out / "build_events.jsonl", build_events)

    summary = {
        "subject": plan.subject_id,
        "category": plan.category,
        "seed_query_count": len(seeds.seed_queries),
        "retrieved_passage_count": len(materials),
        "extraction_passage_count": min(len(materials), args.max_extraction_passages),
        "raw_candidate_count": len(raw_candidates),
        "filtered_candidate_count": len(kept),
        "rejected_candidate_count": len(rejected),
        "grounded_supported_count": sum(1 for item in grounded if item.grounding_status == "SUPPORTED"),
        "grounded_partial_count": sum(1 for item in grounded if item.grounding_status == "PARTIAL"),
        "grounded_unsupported_count": sum(1 for item in grounded if item.grounding_status == "UNSUPPORTED"),
        "candidate_pair_count": len(pairs),
        "candidate_pair_judgement_count": len(pair_judge_logs),
        "candidate_pair_same_count": sum(1 for item in pairs if item.relation == "SAME"),
        "candidate_pair_related_count": sum(1 for item in pairs if item.relation == "RELATED"),
        "candidate_pair_confusable_count": sum(1 for item in pairs if item.relation == "CONFUSABLE"),
        "candidate_pair_distinct_count": sum(1 for item in pairs if item.relation == "DISTINCT"),
        "concept_count": len(registry),
        "active_concept_count": sum(1 for item in registry if item.status == "ACTIVE"),
        "provisional_concept_count": sum(1 for item in registry if item.status == "provisional"),
        "evidence_count": len(evidence),
        "relation_count": len(relations),
        "usage_job_count": len(usage_jobs),
        "usage_material_count": len(usage_materials),
        "usage_card_candidate_count": len(usage_card_candidates),
        "usage_claim_count": len(usage_claims),
        "active_card_count": len(consolidated_cards),
        "rejected_item_count": len(rejected_items),
        "usage_index_count": len(usage_index_rows),
        "usage_faiss_index_count": usage_index_meta.count if usage_index_meta else 0,
        "usage_faiss_dimension": usage_index_meta.dimension if usage_index_meta else None,
        "usage_faiss_path": usage_index_meta.faiss_path if usage_index_meta else None,
        "evidence_section_count": evidence_section_count,
        "model": args.model,
        "model_extraction_request_count": len(extraction_logs),
        "usage_model_extraction_request_count": len(usage_extraction_logs),
        "model_cached_result_count": sum(1 for row in [*extraction_logs, *usage_extraction_logs] if row.get("cached")),
        "model_network_calls": client.network_call_count,
        "model_usage": client.usage_summary(),
        "latency_s": time.perf_counter() - started,
        "note": "Only Wikipag/Wikipedia corpus passages are used for concept and usage-card construction.",
    }
    manifest = build_bank_manifest(
        args=args,
        plan=plan,
        registry=registry,
        usage_cards=consolidated_cards,
        rejected_items=rejected_items,
        evidence_section_count=evidence_section_count,
        usage_index_meta=usage_index_meta,
    )
    (out / "bank_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_report(summary, registry, rejected, consolidated_cards), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a subject-driven, corpus-driven Concept construction smoke.")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--category", default=None)
    parser.add_argument("--subject-source-jsonl", default="data/subject_source/en_subjects.jsonl")
    parser.add_argument("--output-dir", default="runs/smoke_001/subject_concept_smoke_microeconomics")
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
    parser.add_argument("--bank-version", default="smoke-v0.1")
    parser.add_argument("--wikipag-service-url", default="http://127.0.0.1:8897")
    parser.add_argument("--retrieval-timeout-s", type=float, default=120.0)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--model-timeout-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    summary = asyncio.run(run_pipeline(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
