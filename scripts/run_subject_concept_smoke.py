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
from typing import Any, Literal

import httpx
import numpy as np
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ca_mem.embedding import HashingTextEmbedder
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

CURATED_TOPIC_ANCHORS: dict[str, list[str]] = {
    "high_school_microeconomics": [
        "consumer theory",
        "producer theory",
        "supply and demand",
        "market equilibrium",
        "price elasticity of demand",
        "opportunity cost",
        "production cost",
        "market structure",
        "externality",
        "marginal cost and marginal benefit",
        "comparative advantage",
    ],
    "high_school_macroeconomics": [
        "gross domestic product",
        "inflation",
        "unemployment",
        "aggregate demand and aggregate supply",
        "monetary policy",
        "fiscal policy",
        "business cycle",
    ],
    "high_school_biology": [
        "cell biology",
        "genetics",
        "evolution",
        "photosynthesis",
        "cellular respiration",
        "ecology",
    ],
    "conceptual_physics": [
        "Newtonian mechanics",
        "energy conservation",
        "momentum",
        "waves",
        "electric circuits",
        "thermodynamics",
    ],
    "high_school_chemistry": [
        "atomic structure",
        "chemical bonding",
        "stoichiometry",
        "chemical equilibrium",
        "acid base chemistry",
        "thermochemistry",
    ],
}

SlotName = Literal["definition", "trigger", "rule", "pitfall"]


class SubjectProfile(BaseModel):
    subject: str
    category: str
    domain: str
    topic_anchors: list[str]
    excluded_scope: list[str]
    source: str


class ConceptQueryRow(BaseModel):
    query_id: str
    subject: str
    anchor: str
    purpose: str
    query: str


class ConceptPassageRow(BaseModel):
    source_id: str
    subject: str
    query_id: str
    retrieval_query: str
    title: str | None = None
    section: str | None = None
    passage_id: str
    text: str
    retrieval_score: float
    material_score: float
    rank: int


class CandidateConceptDraft(BaseModel):
    raw_name: str
    proposed_name: str
    concept_type: str
    definition_candidate: str
    evidence_span: str
    aliases: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class CandidateExtractionPayload(BaseModel):
    candidates: list[CandidateConceptDraft] = Field(default_factory=list)


class CandidateConceptRow(BaseModel):
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


class FilteredCandidateRow(CandidateConceptRow):
    filter_status: str
    filter_reasons: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    evidence_id: str
    slot: SlotName
    source_id: str
    passage_id: str
    title: str | None = None
    section: str | None = None
    text: str
    evidence_span: str
    retrieval_query: str
    retrieval_score: float
    rank: int


class ConceptRegistryRow(BaseModel):
    concept_id: str
    subject: str
    canonical_name: str
    normalized_name: str
    aliases: list[str] = Field(default_factory=list)
    definition: str
    concept_type: str
    status: str
    confidence: float
    source_ids: list[str] = Field(default_factory=list)
    version: int = 1


class ConceptEvidenceRow(BaseModel):
    concept_id: str
    evidence_id: str
    source_id: str
    passage_id: str
    title: str | None = None
    section: str | None = None
    evidence_span: str
    evidence_type: SlotName = "definition"


class MergeRedirectRow(BaseModel):
    from_candidate_id: str
    to_concept_id: str
    relation: str = "SAME"
    confidence: float


class EvidencePackRow(BaseModel):
    evidence_pack_id: str
    subject: str
    concept_id: str
    concept: str
    definition: str
    evidence: list[EvidenceItem]


class CardSlotDraft(BaseModel):
    text: str
    source_ids: list[str] = Field(default_factory=list)


class RuntimeCardDraftPayload(BaseModel):
    definition: CardSlotDraft
    trigger: list[CardSlotDraft] = Field(default_factory=list)
    rule: list[CardSlotDraft] = Field(default_factory=list)
    pitfall: list[CardSlotDraft] = Field(default_factory=list)


class RuntimeCardRawRow(BaseModel):
    card_id: str
    subject: str
    concept_id: str
    concept: str
    definition: CardSlotDraft
    trigger: list[CardSlotDraft]
    rule: list[CardSlotDraft]
    pitfall: list[CardSlotDraft]
    evidence_pack_id: str
    status: str = "raw"


class RuntimeCardClaimRow(BaseModel):
    claim_id: str
    card_id: str
    concept_id: str
    slot: SlotName
    text: str
    supporting_source_ids: list[str] = Field(default_factory=list)
    decision: str
    reason: str


class RuntimeCardRow(BaseModel):
    card_id: str
    subject: str
    concept_id: str
    concept: str
    definition: str
    trigger: list[str] = Field(default_factory=list)
    rule: list[str] = Field(default_factory=list)
    pitfall: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    status: str = "active"
    version: int = 1


class RuntimeCardIndexRow(BaseModel):
    card_id: str
    subject: str
    concept_id: str
    concept: str
    index_text: str
    payload: dict[str, Any]
    status: str


class RejectedItemRow(BaseModel):
    item_id: str
    item_type: str
    stage: str
    reason: str
    payload: dict[str, Any] = Field(default_factory=dict)


class BuildEventRow(BaseModel):
    event_id: str
    stage: str
    item_id: str
    status: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


def normalize_concept_name(value: str) -> str:
    text = value.lower().strip()
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"[‐‑‒–—-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]+", "", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) > 1 and words[-1].endswith("s") and not words[-1].endswith(("ss", "sis", "us", "ics")):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def subject_label(subject: str) -> str:
    return subject.replace("_", " ").strip()


def subject_prefix(subject: str) -> str:
    tokens = [tok for tok in subject.split("_") if tok not in {"high", "school", "college", "professional"}]
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
            return str(row.get("subject_category") or row.get("category") or DEFAULT_CATEGORY).lower().replace(" ", "_")
    return DEFAULT_CATEGORY


def build_subject_profile(args: argparse.Namespace) -> SubjectProfile:
    category = args.category or load_subject_category(Path(args.subject_source_jsonl), args.subject)
    label = subject_label(args.subject)
    curated = CURATED_TOPIC_ANCHORS.get(args.subject, [])
    anchors = curated or [
        label,
        f"{label} core concepts",
        f"{label} principles",
        f"{label} methods",
        f"{label} terminology",
    ]
    anchors = list(dict.fromkeys(anchor.strip() for anchor in anchors if anchor.strip()))[: args.max_topic_anchors]
    return SubjectProfile(
        subject=args.subject,
        category=category,
        domain=label,
        topic_anchors=anchors,
        excluded_scope=[
            "biographical trivia",
            "historical chronology without reusable subject rule",
            "lists without definitions",
            "reference sections",
            "isolated examples without a general concept",
        ],
        source="curated" if curated else "subject_label_fallback",
    )


def build_concept_queries(profile: SubjectProfile, *, max_queries: int) -> list[ConceptQueryRow]:
    rows: list[ConceptQueryRow] = []
    templates = [
        ("concept_discovery", "{anchor} core concepts {domain}"),
        ("definition_dense", "{anchor} definition principles {domain}"),
        ("taxonomy", "{anchor} types classification terminology {domain}"),
    ]
    seen: set[str] = set()
    for anchor in profile.topic_anchors:
        for purpose, template in templates:
            query = re.sub(r"\s+", " ", template.format(anchor=anchor, domain=profile.domain)).strip()
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                ConceptQueryRow(
                    query_id=stable_id("concept_query", profile.subject, purpose, query),
                    subject=profile.subject,
                    anchor=anchor,
                    purpose=purpose,
                    query=query,
                )
            )
            if len(rows) >= max_queries:
                return rows
    return rows


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


def material_quality_score(text: str, title: str | None, section: str | None, profile: SubjectProfile) -> float:
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
        " method ",
        " process ",
        " in economics",
        " in biology",
        " in physics",
        " in chemistry",
    ]
    score += sum(1.0 for pattern in definition_patterns if pattern in lowered)
    subject_terms = [tok for tok in profile.domain.split() if tok not in {"high", "school"}]
    score += sum(0.6 for term in subject_terms if term in lowered)
    anchor_terms = {tok for anchor in profile.topic_anchors for tok in normalize_concept_name(anchor).split() if len(tok) > 3}
    score += min(4.0, sum(0.25 for term in anchor_terms if term in lowered))
    if title and any(term in title.lower() for term in anchor_terms):
        score += 1.0
    if section and section.lower() in {"lead", "introduction", "overview"}:
        score += 0.3
    return score


def should_keep_passage(hit: dict[str, Any], profile: SubjectProfile) -> bool:
    if hit.get("missing"):
        return False
    text = str(hit.get("text") or "").strip()
    if len(text) < 220:
        return False
    title = str(hit.get("title") or "")
    section = str(hit.get("section") or "")
    section_l = section.lower()
    if any(term == section_l or term in section_l for term in BAD_SECTION_TERMS):
        return False
    if title.lower().endswith("(disambiguation)"):
        return False
    return material_quality_score(text, title, section, profile) >= 1.0


def retrieve_concept_passages(
    profile: SubjectProfile,
    queries: list[ConceptQueryRow],
    *,
    service_url: str,
    top_k_per_query: int,
    timeout_s: float,
    max_passages: int,
) -> list[ConceptPassageRow]:
    batch_results = search_wiki_batch(service_url, [row.query for row in queries], top_k=top_k_per_query, timeout_s=timeout_s)
    query_by_text = {row.query: row for row in queries}
    by_passage: dict[str, ConceptPassageRow] = {}
    for item in batch_results:
        query_text = str(item.get("query") or "")
        query_row = query_by_text.get(query_text)
        if query_row is None:
            continue
        for hit in item.get("results", []):
            if not isinstance(hit, dict) or not should_keep_passage(hit, profile):
                continue
            passage_id = str(hit.get("passage_id") or "")
            if not passage_id:
                continue
            text = str(hit.get("text") or "")
            title = str(hit.get("title")) if hit.get("title") is not None else None
            section = str(hit.get("section")) if hit.get("section") is not None else None
            row = ConceptPassageRow(
                source_id=stable_id("source", profile.subject, passage_id),
                subject=profile.subject,
                query_id=query_row.query_id,
                retrieval_query=query_text,
                title=title,
                section=section,
                passage_id=passage_id,
                text=text,
                retrieval_score=float(hit.get("score") or 0.0),
                material_score=material_quality_score(text, title, section, profile),
                rank=int(hit.get("rank") or 0),
            )
            previous = by_passage.get(passage_id)
            if previous is None or (row.material_score + row.retrieval_score) > (previous.material_score + previous.retrieval_score):
                by_passage[passage_id] = row
    return sorted(
        by_passage.values(),
        key=lambda row: (-row.material_score, -row.retrieval_score, row.rank, row.passage_id),
    )[:max_passages]


def candidate_extraction_prompt(profile: SubjectProfile, passage: ConceptPassageRow) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Extract reusable academic concepts from English knowledge material. "
                "The pipeline is subject-driven and corpus-driven. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": f"""Subject: {profile.subject}
Domain: {profile.domain}
Topic anchors: {", ".join(profile.topic_anchors)}
Excluded scope: {", ".join(profile.excluded_scope)}

Source title: {passage.title or ""}
Source section: {passage.section or ""}

Passage:
{passage.text[:4500]}

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
      "concept_type": "principle|metric|model|method|process|classification|defined_term|other",
      "definition_candidate": "one sentence definition",
      "evidence_span": "short span from passage",
      "aliases": ["optional aliases"],
      "confidence": 0.0
    }}
  ]
}}""",
        },
    ]


def validate_candidate_payload(payload: dict[str, Any]) -> CandidateExtractionPayload:
    parsed = CandidateExtractionPayload.model_validate(payload)
    parsed.candidates = parsed.candidates[:5]
    return parsed


async def extract_candidate_concepts(
    profile: SubjectProfile,
    passages: list[ConceptPassageRow],
    *,
    client: SmokeModelClient,
    max_passages: int,
) -> tuple[list[CandidateConceptRow], list[dict[str, Any]]]:
    selected = passages[:max_passages]
    tasks = [
        client.complete_json(
            candidate_extraction_prompt(profile, passage),
            namespace=f"wiki_concept.extract.{profile.subject}.{passage.source_id}",
            validate=validate_candidate_payload,
        )
        for passage in selected
    ]
    outputs = await asyncio.gather(*tasks) if tasks else []
    candidates: list[CandidateConceptRow] = []
    logs: list[dict[str, Any]] = []
    seen_key_counts: Counter[str] = Counter()
    for passage, (payload, result, error) in zip(selected, outputs):
        logs.append(
            {
                "source_id": passage.source_id,
                "ok": isinstance(payload, CandidateExtractionPayload),
                "error": error,
                "latency_s": result.latency_s if result else None,
                "cached": result.cached if result else None,
                "usage": result.usage if result else {},
            }
        )
        if not isinstance(payload, CandidateExtractionPayload):
            continue
        for draft in payload.candidates:
            proposed = draft.proposed_name.strip() or draft.raw_name.strip()
            normalized = normalize_concept_name(proposed)
            if not normalized:
                continue
            base_key = stable_id("cand", profile.subject, passage.source_id, normalized)
            seen_key_counts[base_key] += 1
            candidates.append(
                CandidateConceptRow(
                    candidate_id=f"{base_key}_{seen_key_counts[base_key]}",
                    subject=profile.subject,
                    raw_name=draft.raw_name.strip(),
                    proposed_name=proposed,
                    normalized_name=normalized,
                    concept_type=draft.concept_type.strip() or "other",
                    definition_candidate=draft.definition_candidate.strip(),
                    source_id=passage.source_id,
                    passage_id=passage.passage_id,
                    title=passage.title,
                    section=passage.section,
                    evidence_span=draft.evidence_span.strip(),
                    aliases=[alias.strip() for alias in draft.aliases if alias.strip()],
                    confidence=float(draft.confidence or 0.0),
                )
            )
    return candidates, logs


def filter_candidate(candidate: CandidateConceptRow) -> FilteredCandidateRow:
    reasons: list[str] = []
    name = candidate.normalized_name
    words = name.split()
    if not name:
        reasons.append("EMPTY_NAME")
    if name in TOO_BROAD_NAMES:
        reasons.append("TOO_BROAD")
    if len(words) > 7:
        reasons.append("TOO_NARROW")
    if candidate.confidence < 0.45:
        reasons.append("LOW_CONFIDENCE")
    if len(candidate.definition_candidate) < 28:
        reasons.append("WEAK_DEFINITION")
    if len(candidate.evidence_span) < 16 and len(candidate.definition_candidate) < 60:
        reasons.append("WEAK_EVIDENCE_SPAN")
    if name.count(" and ") >= 1 and len(words) >= 5:
        reasons.append("COMPOSITE_RISK")
    status = "IS_CONCEPT" if not reasons else reasons[0]
    return FilteredCandidateRow(**candidate.model_dump(mode="json"), filter_status=status, filter_reasons=reasons)


def filter_candidates(candidates: list[CandidateConceptRow]) -> tuple[list[FilteredCandidateRow], list[FilteredCandidateRow]]:
    rows = [filter_candidate(candidate) for candidate in candidates]
    return [row for row in rows if row.filter_status == "IS_CONCEPT"], [row for row in rows if row.filter_status != "IS_CONCEPT"]


def find_evidence_span(text: str, name: str, fallback: str) -> str:
    tokens = [tok for tok in normalize_concept_name(name).split() if tok]
    if tokens:
        pattern = re.compile(r"[^.?!]*(?:" + "|".join(map(re.escape, tokens[:3])) + r")[^.?!]*[.?!]", re.I)
        match = pattern.search(text)
        if match:
            return re.sub(r"\s+", " ", match.group(0)).strip()[:700]
    return re.sub(r"\s+", " ", fallback or text[:700]).strip()[:700]


def name_supported_by_hit(candidate: FilteredCandidateRow, hit: dict[str, Any]) -> bool:
    haystack = normalize_concept_name(" ".join([str(hit.get("title") or ""), str(hit.get("section") or ""), str(hit.get("text") or "")]))
    tokens = [tok for tok in candidate.normalized_name.split() if len(tok) > 2]
    if not tokens:
        return False
    hits = sum(1 for tok in tokens if tok in haystack)
    return hits >= min(len(tokens), 2)


def semantic_same_signature(name: str) -> str:
    tokens = [
        tok
        for tok in normalize_concept_name(name).split()
        if tok not in {"theory", "of", "model", "concept", "principle"}
    ]
    if len(tokens) < 2:
        return ""
    return " ".join(sorted(tokens))


def ground_and_dedup_concepts(
    profile: SubjectProfile,
    candidates: list[FilteredCandidateRow],
    *,
    service_url: str,
    top_k: int,
    timeout_s: float,
    target_active: int,
) -> tuple[list[ConceptRegistryRow], list[ConceptEvidenceRow], list[MergeRedirectRow], list[dict[str, Any]], list[RejectedItemRow]]:
    if not candidates:
        return [], [], [], [], []
    queries = [f"{candidate.proposed_name} definition {profile.domain}" for candidate in candidates]
    batch_results = search_wiki_batch(service_url, queries, top_k=top_k, timeout_s=timeout_s)
    grounded_rows: list[tuple[FilteredCandidateRow, EvidenceItem | None, str, float]] = []
    grounding_logs: list[dict[str, Any]] = []
    rejected: list[RejectedItemRow] = []
    for candidate, item in zip(candidates, batch_results):
        evidence_item: EvidenceItem | None = None
        for hit in item.get("results", []):
            if not isinstance(hit, dict) or hit.get("missing"):
                continue
            grounding_logs.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "query": item.get("query"),
                    "rank": hit.get("rank"),
                    "score": hit.get("score"),
                    "passage_id": hit.get("passage_id"),
                    "title": hit.get("title"),
                    "section": hit.get("section"),
                }
            )
            if name_supported_by_hit(candidate, hit):
                passage_id = str(hit.get("passage_id") or "")
                source_id = stable_id("source", profile.subject, passage_id)
                evidence_item = EvidenceItem(
                    evidence_id=stable_id("evidence", candidate.candidate_id, "definition", source_id),
                    slot="definition",
                    source_id=source_id,
                    passage_id=passage_id,
                    title=str(hit.get("title")) if hit.get("title") is not None else None,
                    section=str(hit.get("section")) if hit.get("section") is not None else None,
                    text=str(hit.get("text") or ""),
                    evidence_span=find_evidence_span(str(hit.get("text") or ""), candidate.proposed_name, candidate.evidence_span),
                    retrieval_query=str(item.get("query") or ""),
                    retrieval_score=float(hit.get("score") or 0.0),
                    rank=int(hit.get("rank") or 0),
                )
                break
        if evidence_item is None:
            rejected.append(
                RejectedItemRow(
                    item_id=candidate.candidate_id,
                    item_type="concept_candidate",
                    stage="concept_grounding",
                    reason="UNSUPPORTED",
                    payload=candidate.model_dump(mode="json"),
                )
            )
            continue
        grounded_rows.append((candidate, evidence_item, "SUPPORTED", min(0.99, max(candidate.confidence, 0.72))))

    by_signature: dict[str, list[tuple[FilteredCandidateRow, EvidenceItem, str, float]]] = defaultdict(list)
    for candidate, evidence_item, status, confidence in grounded_rows:
        key = candidate.normalized_name or semantic_same_signature(candidate.normalized_name)
        by_signature[key].append((candidate, evidence_item, status, confidence))

    registry: list[ConceptRegistryRow] = []
    evidence_rows: list[ConceptEvidenceRow] = []
    redirects: list[MergeRedirectRow] = []
    used_ids: Counter[str] = Counter()
    prefix = subject_prefix(profile.subject)
    sorted_clusters = sorted(
        by_signature.values(),
        key=lambda items: (-max(row[3] for row in items), min(row[0].normalized_name for row in items)),
    )
    for items in sorted_clusters:
        if len(registry) >= target_active:
            break
        best = sorted(items, key=lambda row: (-row[3], row[0].normalized_name))[0]
        candidate = best[0]
        base_id = f"{prefix}-{slug_name(candidate.normalized_name)}"
        used_ids[base_id] += 1
        concept_id = base_id if used_ids[base_id] == 1 else f"{base_id}-{used_ids[base_id]}"
        aliases = sorted(
            {
                alias
                for cand, _ev, _status, _conf in items
                for alias in [cand.raw_name, cand.proposed_name, *cand.aliases]
                if normalize_concept_name(alias) != candidate.normalized_name
            },
            key=str.lower,
        )[:8]
        source_ids = sorted({ev.source_id for _cand, ev, _status, _conf in items})
        registry.append(
            ConceptRegistryRow(
                concept_id=concept_id,
                subject=profile.subject,
                canonical_name=candidate.normalized_name,
                normalized_name=candidate.normalized_name,
                aliases=aliases,
                definition=candidate.definition_candidate,
                concept_type=candidate.concept_type,
                status="ACTIVE",
                confidence=max(conf for _cand, _ev, _status, conf in items),
                source_ids=source_ids,
            )
        )
        for cand, ev, _status, conf in items:
            redirects.append(MergeRedirectRow(from_candidate_id=cand.candidate_id, to_concept_id=concept_id, confidence=conf))
            evidence_rows.append(
                ConceptEvidenceRow(
                    concept_id=concept_id,
                    evidence_id=ev.evidence_id,
                    source_id=ev.source_id,
                    passage_id=ev.passage_id,
                    title=ev.title,
                    section=ev.section,
                    evidence_span=ev.evidence_span,
                    evidence_type="definition",
                )
            )
    return registry, evidence_rows, redirects, grounding_logs, rejected


def evidence_queries(profile: SubjectProfile, concept: ConceptRegistryRow) -> list[tuple[SlotName, str]]:
    name = concept.canonical_name
    domain = profile.domain
    return [
        ("definition", f"{name} definition {domain}"),
        ("trigger", f"{name} when to use application conditions {domain}"),
        ("rule", f"{name} rule formula procedure decision {domain}"),
        ("pitfall", f"{name} common mistake misconception exception boundary {domain}"),
    ]


def build_evidence_packs(
    profile: SubjectProfile,
    registry: list[ConceptRegistryRow],
    *,
    service_url: str,
    top_k: int,
    timeout_s: float,
    max_concepts: int,
    max_items_per_slot: int,
) -> list[EvidencePackRow]:
    concepts = [row for row in registry if row.status == "ACTIVE"][:max_concepts]
    all_queries: list[str] = []
    query_meta: list[tuple[ConceptRegistryRow, SlotName, str]] = []
    for concept in concepts:
        for slot, query in evidence_queries(profile, concept):
            all_queries.append(query)
            query_meta.append((concept, slot, query))
    batch_results = search_wiki_batch(service_url, all_queries, top_k=top_k, timeout_s=timeout_s) if all_queries else []
    evidence_by_concept: dict[str, list[EvidenceItem]] = defaultdict(list)
    for (concept, slot, query), item in zip(query_meta, batch_results):
        kept = 0
        for hit in item.get("results", []):
            if kept >= max_items_per_slot:
                break
            if not isinstance(hit, dict) or hit.get("missing"):
                continue
            text = str(hit.get("text") or "").strip()
            if len(text) < 160:
                continue
            passage_id = str(hit.get("passage_id") or "")
            if not passage_id:
                continue
            normalized_text = normalize_concept_name(" ".join([str(hit.get("title") or ""), str(hit.get("section") or ""), text]))
            name_tokens = [tok for tok in concept.normalized_name.split() if len(tok) > 2]
            if name_tokens and sum(1 for tok in name_tokens if tok in normalized_text) < min(len(name_tokens), 2):
                continue
            source_id = stable_id("source", profile.subject, passage_id)
            evidence_by_concept[concept.concept_id].append(
                EvidenceItem(
                    evidence_id=stable_id("evidence", concept.concept_id, slot, source_id, kept),
                    slot=slot,
                    source_id=source_id,
                    passage_id=passage_id,
                    title=str(hit.get("title")) if hit.get("title") is not None else None,
                    section=str(hit.get("section")) if hit.get("section") is not None else None,
                    text=text,
                    evidence_span=find_evidence_span(text, concept.canonical_name, text[:700]),
                    retrieval_query=query,
                    retrieval_score=float(hit.get("score") or 0.0),
                    rank=int(hit.get("rank") or 0),
                )
            )
            kept += 1
    return [
        EvidencePackRow(
            evidence_pack_id=stable_id("evidence_pack", profile.subject, concept.concept_id),
            subject=profile.subject,
            concept_id=concept.concept_id,
            concept=concept.canonical_name,
            definition=concept.definition,
            evidence=evidence_by_concept.get(concept.concept_id, []),
        )
        for concept in concepts
    ]


def compact_card_prompt(pack: EvidencePackRow) -> list[dict[str, str]]:
    evidence_block = "\n\n".join(
        [
            f"Source ID: {ev.source_id}\n"
            f"Slot target: {ev.slot}\n"
            f"Title: {ev.title or ''}\n"
            f"Section: {ev.section or ''}\n"
            f"Evidence: {ev.evidence_span or ev.text[:900]}"
            for ev in pack.evidence[:16]
        ]
    )
    return [
        {
            "role": "system",
            "content": (
                "Build one compact runtime concept card from supplied evidence only. "
                "Only produce definition, trigger, rule, and pitfall slots. Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": f"""Subject: {pack.subject}
Concept ID: {pack.concept_id}
Concept: {pack.concept}
Initial definition: {pack.definition}

Evidence pack:
{evidence_block}

Create exactly one compact card for this concept.

Rules:
- Use only evidence above.
- Keep the card short.
- definition: one concise sentence.
- trigger: 1 to 4 conditions for when to use this concept.
- rule: 1 to 5 procedural rules.
- pitfall: 0 to 4 common mistakes or boundaries.
- Every slot item must include source_ids from the evidence pack.
- If evidence is insufficient, still return JSON but use empty arrays for unsupported trigger/rule/pitfall.

Return JSON:
{{
  "definition": {{"text": "...", "source_ids": ["source_..."]}},
  "trigger": [{{"text": "...", "source_ids": ["source_..."]}}],
  "rule": [{{"text": "...", "source_ids": ["source_..."]}}],
  "pitfall": [{{"text": "...", "source_ids": ["source_..."]}}]
}}""",
        },
    ]


def validate_runtime_card_payload(payload: dict[str, Any]) -> RuntimeCardDraftPayload:
    parsed = RuntimeCardDraftPayload.model_validate(payload)
    parsed.trigger = parsed.trigger[:4]
    parsed.rule = parsed.rule[:5]
    parsed.pitfall = parsed.pitfall[:4]
    return parsed


async def extract_runtime_cards(
    packs: list[EvidencePackRow],
    *,
    client: SmokeModelClient,
) -> tuple[list[RuntimeCardRawRow], list[dict[str, Any]]]:
    runnable = [pack for pack in packs if pack.evidence]
    tasks = [
        client.complete_json(
            compact_card_prompt(pack),
            namespace=f"wiki_runtime_card.extract.{pack.subject}.{pack.concept_id}",
            validate=validate_runtime_card_payload,
        )
        for pack in runnable
    ]
    outputs = await asyncio.gather(*tasks) if tasks else []
    cards: list[RuntimeCardRawRow] = []
    logs: list[dict[str, Any]] = []
    for pack, (payload, result, error) in zip(runnable, outputs):
        logs.append(
            {
                "evidence_pack_id": pack.evidence_pack_id,
                "concept_id": pack.concept_id,
                "ok": isinstance(payload, RuntimeCardDraftPayload),
                "error": error,
                "latency_s": result.latency_s if result else None,
                "cached": result.cached if result else None,
                "usage": result.usage if result else {},
            }
        )
        if not isinstance(payload, RuntimeCardDraftPayload):
            continue
        cards.append(
            RuntimeCardRawRow(
                card_id=stable_id("runtime_card", pack.subject, pack.concept_id),
                subject=pack.subject,
                concept_id=pack.concept_id,
                concept=pack.concept,
                definition=payload.definition,
                trigger=payload.trigger,
                rule=payload.rule,
                pitfall=payload.pitfall,
                evidence_pack_id=pack.evidence_pack_id,
            )
        )
    return cards, logs


TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def content_tokens(text: str) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "when",
        "where",
        "use",
        "used",
        "concept",
        "rule",
    }
    return {tok for tok in TOKEN_RE.findall(text.lower()) if tok not in stop}


def verify_slot(slot_text: str, source_ids: list[str], evidence_by_source: dict[str, list[EvidenceItem]]) -> tuple[str, list[str], str]:
    text = re.sub(r"\s+", " ", slot_text).strip()
    valid_source_ids = [source_id for source_id in dict.fromkeys(source_ids) if source_id in evidence_by_source]
    if len(text) < 10:
        return "REJECT", [], "too_short"
    if not valid_source_ids:
        return "REJECT", [], "missing_valid_source"
    claim_tokens = content_tokens(text)
    if not claim_tokens:
        return "REJECT", [], "no_content_tokens"
    supported: list[str] = []
    for source_id in valid_source_ids:
        evidence_text = " ".join(ev.evidence_span or ev.text for ev in evidence_by_source[source_id])
        evidence_tokens = content_tokens(evidence_text)
        overlap = claim_tokens & evidence_tokens
        # Definition/compact procedural claims often paraphrase. Require a light
        # lexical anchor plus explicit source provenance; this is a minimal gate.
        if len(overlap) >= max(1, min(3, len(claim_tokens) // 4)):
            supported.append(source_id)
    if supported:
        return "ACCEPT", supported, "source_overlap"
    return "REJECT", [], "insufficient_overlap"


def verify_runtime_cards(
    raw_cards: list[RuntimeCardRawRow],
    packs: list[EvidencePackRow],
) -> tuple[list[RuntimeCardRow], list[RuntimeCardClaimRow], list[RejectedItemRow]]:
    pack_by_id = {pack.evidence_pack_id: pack for pack in packs}
    final_cards: list[RuntimeCardRow] = []
    claims: list[RuntimeCardClaimRow] = []
    rejected: list[RejectedItemRow] = []
    for raw in raw_cards:
        pack = pack_by_id.get(raw.evidence_pack_id)
        if pack is None:
            rejected.append(
                RejectedItemRow(
                    item_id=raw.card_id,
                    item_type="runtime_card",
                    stage="card_verification",
                    reason="missing_evidence_pack",
                    payload=raw.model_dump(mode="json"),
                )
            )
            continue
        evidence_by_source: dict[str, list[EvidenceItem]] = defaultdict(list)
        for ev in pack.evidence:
            evidence_by_source[ev.source_id].append(ev)

        accepted: dict[SlotName, list[str]] = {"definition": [], "trigger": [], "rule": [], "pitfall": []}
        accepted_sources: set[str] = set()
        slot_items: list[tuple[SlotName, CardSlotDraft]] = [
            ("definition", raw.definition),
            *[("trigger", item) for item in raw.trigger],
            *[("rule", item) for item in raw.rule],
            *[("pitfall", item) for item in raw.pitfall],
        ]
        for idx, (slot, item) in enumerate(slot_items, start=1):
            decision, supporting_source_ids, reason = verify_slot(item.text, item.source_ids, evidence_by_source)
            if decision == "ACCEPT":
                text = re.sub(r"\s+", " ", item.text).strip()
                if text not in accepted[slot]:
                    accepted[slot].append(text)
                accepted_sources.update(supporting_source_ids)
            claims.append(
                RuntimeCardClaimRow(
                    claim_id=stable_id("runtime_claim", raw.card_id, idx, slot, item.text),
                    card_id=raw.card_id,
                    concept_id=raw.concept_id,
                    slot=slot,
                    text=re.sub(r"\s+", " ", item.text).strip(),
                    supporting_source_ids=supporting_source_ids,
                    decision=decision,
                    reason=reason,
                )
            )

        if not accepted["definition"] or not accepted["rule"]:
            rejected.append(
                RejectedItemRow(
                    item_id=raw.card_id,
                    item_type="runtime_card",
                    stage="card_verification",
                    reason="missing_supported_definition_or_rule",
                    payload=raw.model_dump(mode="json"),
                )
            )
            continue
        final_cards.append(
            RuntimeCardRow(
                card_id=raw.card_id,
                subject=raw.subject,
                concept_id=raw.concept_id,
                concept=raw.concept,
                definition=accepted["definition"][0],
                trigger=accepted["trigger"],
                rule=accepted["rule"],
                pitfall=accepted["pitfall"],
                source_ids=sorted(accepted_sources),
            )
        )
    return final_cards, claims, rejected


def render_runtime_card(card: RuntimeCardRow) -> str:
    sections = [
        f"Subject: {card.subject}",
        f"Concept: {card.concept}",
        f"Definition: {card.definition}",
    ]
    if card.trigger:
        sections.append("Trigger:\n" + "\n".join(f"- {item}" for item in card.trigger))
    if card.rule:
        sections.append("Rule:\n" + "\n".join(f"- {item}" for item in card.rule))
    if card.pitfall:
        sections.append("Pitfall:\n" + "\n".join(f"- {item}" for item in card.pitfall))
    return "\n".join(sections)


def build_runtime_card_index(cards: list[RuntimeCardRow], out: Path) -> tuple[list[RuntimeCardIndexRow], dict[str, Any]]:
    rows = [
        RuntimeCardIndexRow(
            card_id=card.card_id,
            subject=card.subject,
            concept_id=card.concept_id,
            concept=card.concept,
            index_text=render_runtime_card(card),
            payload=card.model_dump(mode="json"),
            status=card.status,
        )
        for card in cards
    ]
    embedder = HashingTextEmbedder()
    matrix = embedder.embed([row.index_text for row in rows])
    np.save(out / "runtime_card_index.npy", matrix)
    meta = {
        "index_type": "numpy_dense_matrix",
        "metric": "cosine_on_normalized_embeddings",
        "embedding_backend": "hash",
        "embedding_model": embedder.model_name,
        "dimension": int(matrix.shape[1]) if matrix.ndim == 2 else embedder.dim,
        "count": len(rows),
        "matrix_path": str(out / "runtime_card_index.npy"),
        "ids_path": str(out / "runtime_card_index.jsonl"),
    }
    (out / "runtime_card_index_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows, meta


def write_evidence_sections_parquet(out: Path, concept_passages: list[ConceptPassageRow], packs: list[EvidencePackRow]) -> None:
    rows: dict[str, dict[str, Any]] = {}
    for passage in concept_passages:
        rows[passage.source_id] = {
            "source_id": passage.source_id,
            "source_kind": "concept_discovery",
            "subject": passage.subject,
            "passage_id": passage.passage_id,
            "title": passage.title,
            "section": passage.section,
            "text": passage.text,
            "retrieval_query": passage.retrieval_query,
            "retrieval_score": passage.retrieval_score,
        }
    for pack in packs:
        for ev in pack.evidence:
            rows[ev.source_id] = {
                "source_id": ev.source_id,
                "source_kind": f"card_{ev.slot}_evidence",
                "subject": pack.subject,
                "concept_id": pack.concept_id,
                "evidence_pack_id": pack.evidence_pack_id,
                "passage_id": ev.passage_id,
                "title": ev.title,
                "section": ev.section,
                "text": ev.text,
                "retrieval_query": ev.retrieval_query,
                "retrieval_score": ev.retrieval_score,
            }
    import pandas as pd

    pd.DataFrame(list(rows.values())).to_parquet(out / "evidence_sections.parquet", index=False)


def build_manifest(
    *,
    args: argparse.Namespace,
    profile: SubjectProfile,
    registry: list[ConceptRegistryRow],
    evidence_packs: list[EvidencePackRow],
    raw_cards: list[RuntimeCardRawRow],
    runtime_cards: list[RuntimeCardRow],
    claims: list[RuntimeCardClaimRow],
    rejected_items: list[RejectedItemRow],
    index_meta: dict[str, Any],
    evidence_section_count: int,
    model_network_calls: int,
) -> dict[str, Any]:
    return {
        "bank_version": args.bank_version,
        "pipeline_version": "wiki_compact_card_v1",
        "source_snapshot": "wikipedia-en-2026-07-01-sherlock-faiss",
        "subjects": 1,
        "subject_ids": [profile.subject],
        "subject": profile.subject,
        "category": profile.category,
        "concept_count": len(registry),
        "active_concept_count": sum(1 for row in registry if row.status == "ACTIVE"),
        "evidence_pack_count": len(evidence_packs),
        "raw_runtime_card_count": len(raw_cards),
        "runtime_card_count": len(runtime_cards),
        "active_card_count": len(runtime_cards),
        "runtime_card_claim_count": len(claims),
        "accepted_claim_count": sum(1 for row in claims if row.decision == "ACCEPT"),
        "rejected_claim_count": sum(1 for row in claims if row.decision == "REJECT"),
        "evidence_section_count": evidence_section_count,
        "rejected_item_count": len(rejected_items),
        "runtime_card_index": index_meta,
        "construction_model": args.model,
        "model_network_calls": model_network_calls,
        "construction_cutoff": datetime.now(timezone.utc).isoformat(),
        "source_corpus_only": True,
        "one_card_per_subject_concept": True,
        "runtime_card_slots": ["definition", "trigger", "rule", "pitfall"],
    }


def render_report(summary: dict[str, Any], runtime_cards: list[RuntimeCardRow], rejected: list[RejectedItemRow]) -> str:
    lines = [
        "# Wiki Compact Runtime Card Construction",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Runtime cards",
        "",
        "| card_id | concept | trigger | rule | pitfall |",
        "|---|---:|---:|---:|---:|",
    ]
    for card in runtime_cards:
        lines.append(f"| `{card.card_id}` | {card.concept} | {len(card.trigger)} | {len(card.rule)} | {len(card.pitfall)} |")
    lines.extend(["", "## Rejected items", "", "| item_id | stage | reason |", "|---|---|---|"])
    for item in rejected[:50]:
        lines.append(f"| `{item.item_id}` | {item.stage} | {item.reason} |")
    return "\n".join(lines) + "\n"


async def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    profile = build_subject_profile(args)
    write_jsonl(out / "subject_profile.jsonl", [profile])

    concept_queries = build_concept_queries(profile, max_queries=args.max_concept_queries)
    write_jsonl(out / "concept_queries.jsonl", concept_queries)

    concept_passages = retrieve_concept_passages(
        profile,
        concept_queries,
        service_url=args.wikipag_service_url,
        top_k_per_query=args.top_k_per_query,
        timeout_s=args.retrieval_timeout_s,
        max_passages=args.max_passages,
    )
    write_jsonl(out / "concept_passages.jsonl", concept_passages)

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
        candidate_concepts, candidate_logs = await extract_candidate_concepts(
            profile,
            concept_passages,
            client=client,
            max_passages=args.max_extraction_passages,
        )
        write_jsonl(out / "candidate_extraction_logs.jsonl", candidate_logs)
        write_jsonl(out / "candidate_concepts.raw.jsonl", candidate_concepts)

        kept_candidates, rejected_candidates = filter_candidates(candidate_concepts)
        kept_candidates = sorted(kept_candidates, key=lambda row: (-row.confidence, row.normalized_name))[: args.max_grounding_candidates]
        write_jsonl(out / "candidate_concepts.jsonl", kept_candidates)
        write_jsonl(out / "candidate_concepts.rejected.jsonl", rejected_candidates)

        registry, concept_evidence, redirects, grounding_logs, grounding_rejections = ground_and_dedup_concepts(
            profile,
            kept_candidates,
            service_url=args.wikipag_service_url,
            top_k=args.grounding_top_k,
            timeout_s=args.retrieval_timeout_s,
            target_active=args.target_active_concepts,
        )
        write_jsonl(out / "grounding_logs.jsonl", grounding_logs)
        write_jsonl(out / "concept_registry.jsonl", registry)
        write_jsonl(out / "concept_evidence.jsonl", concept_evidence)
        write_jsonl(out / "merge_redirects.jsonl", redirects)

        evidence_packs = build_evidence_packs(
            profile,
            registry,
            service_url=args.wikipag_service_url,
            top_k=args.evidence_top_k,
            timeout_s=args.retrieval_timeout_s,
            max_concepts=args.max_card_concepts,
            max_items_per_slot=args.max_evidence_items_per_slot,
        )
        write_jsonl(out / "evidence_packs.jsonl", evidence_packs)

        raw_cards, card_extraction_logs = await extract_runtime_cards(evidence_packs, client=client)
        write_jsonl(out / "runtime_card_extraction_logs.jsonl", card_extraction_logs)
        write_jsonl(out / "runtime_cards.raw.jsonl", raw_cards)
    finally:
        await client.aclose()

    runtime_cards, claims, card_rejections = verify_runtime_cards(raw_cards, evidence_packs)
    write_jsonl(out / "runtime_card_claims.jsonl", claims)
    write_jsonl(out / "runtime_cards.jsonl", runtime_cards)

    index_rows, index_meta = build_runtime_card_index(runtime_cards, out)
    write_jsonl(out / "runtime_card_index.jsonl", index_rows)

    rejected_items: list[RejectedItemRow] = [
        *[
            RejectedItemRow(
                item_id=item.candidate_id,
                item_type="concept_candidate",
                stage="concept_filter",
                reason=item.filter_status,
                payload=item.model_dump(mode="json"),
            )
            for item in rejected_candidates
        ],
        *grounding_rejections,
        *card_rejections,
    ]
    write_jsonl(out / "rejected_items.jsonl", rejected_items)
    write_evidence_sections_parquet(out, concept_passages, evidence_packs)
    evidence_section_count = len(
        {passage.source_id for passage in concept_passages}
        | {ev.source_id for pack in evidence_packs for ev in pack.evidence}
    )

    build_events = [
        BuildEventRow(
            event_id=stable_id("event", profile.subject, stage),
            stage=stage,
            item_id=profile.subject,
            status="ok",
            message=message,
            payload=payload,
        )
        for stage, message, payload in [
            ("subject_profile", "subject profile written", {"topic_anchor_count": len(profile.topic_anchors)}),
            ("concept_queries", "concept discovery queries written", {"query_count": len(concept_queries)}),
            ("concept_passages", "concept passages retrieved", {"passage_count": len(concept_passages)}),
            ("candidate_concepts", "candidate concepts extracted", {"candidate_count": len(candidate_concepts)}),
            ("concept_registry", "concepts grounded and deduplicated", {"concept_count": len(registry)}),
            ("evidence_packs", "evidence packs retrieved", {"evidence_pack_count": len(evidence_packs)}),
            ("runtime_cards_raw", "compact raw cards extracted", {"raw_card_count": len(raw_cards)}),
            ("runtime_card_claims", "card slots verified", {"claim_count": len(claims)}),
            ("runtime_cards", "final runtime cards written", {"runtime_card_count": len(runtime_cards)}),
            ("runtime_card_index", "runtime card index written", {"index_count": len(index_rows)}),
        ]
    ]
    write_jsonl(out / "build_events.jsonl", build_events)

    summary = {
        "subject": profile.subject,
        "category": profile.category,
        "pipeline_version": "wiki_compact_card_v1",
        "topic_anchor_count": len(profile.topic_anchors),
        "concept_query_count": len(concept_queries),
        "concept_passage_count": len(concept_passages),
        "candidate_concept_count": len(candidate_concepts),
        "kept_candidate_count": len(kept_candidates),
        "rejected_candidate_count": len(rejected_candidates),
        "concept_count": len(registry),
        "active_concept_count": sum(1 for row in registry if row.status == "ACTIVE"),
        "concept_evidence_count": len(concept_evidence),
        "evidence_pack_count": len(evidence_packs),
        "evidence_item_count": sum(len(pack.evidence) for pack in evidence_packs),
        "raw_runtime_card_count": len(raw_cards),
        "runtime_card_claim_count": len(claims),
        "accepted_claim_count": sum(1 for row in claims if row.decision == "ACCEPT"),
        "rejected_claim_count": sum(1 for row in claims if row.decision == "REJECT"),
        "runtime_card_count": len(runtime_cards),
        "active_card_count": len(runtime_cards),
        "runtime_card_index_count": len(index_rows),
        "evidence_section_count": evidence_section_count,
        "rejected_item_count": len(rejected_items),
        "model": args.model,
        "model_network_calls": client.network_call_count,
        "model_usage": client.usage_summary(),
        "latency_s": time.perf_counter() - started,
    }
    manifest = build_manifest(
        args=args,
        profile=profile,
        registry=registry,
        evidence_packs=evidence_packs,
        raw_cards=raw_cards,
        runtime_cards=runtime_cards,
        claims=claims,
        rejected_items=rejected_items,
        index_meta=index_meta,
        evidence_section_count=evidence_section_count,
        model_network_calls=client.network_call_count,
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "bank_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_report(summary, runtime_cards, rejected_items), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run minimal subject-driven Wiki compact-card construction.")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--category", default=None)
    parser.add_argument("--subject-source-jsonl", default="data/subject_source/en_subjects.jsonl")
    parser.add_argument("--output-dir", default="runs/smoke_001/subject_compact_card_smoke")
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
    parser.add_argument("--bank-version", default="wiki-compact-card-v0.1")
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
