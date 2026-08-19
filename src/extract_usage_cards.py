from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from src.smoke_test.config import SmokeTestConfig
from src.smoke_test.io import append_error
from src.smoke_test.model_client import SmokeModelClient
from src.smoke_test.schemas import EvidenceClaim, RetrievalResult, UsageCard, UsageJob
from src.smoke_test.stage_utils import stable_id


def extract_usage_cards(
    jobs: list[UsageJob],
    retrieval_results: list[RetrievalResult],
    config: SmokeTestConfig,
    *,
    dry_run: bool,
    client: SmokeModelClient | None = None,
    errors_path: Path | None = None,
) -> list[UsageCard]:
    if not dry_run:
        if client is None:
            raise ValueError("model-backed usage-card extraction requires SmokeModelClient")
        return asyncio.run(_extract_usage_cards_model(jobs, retrieval_results, config, client=client, errors_path=errors_path))
    retrieval_by_job = {result.usage_job_id: result for result in retrieval_results}
    cards: list[UsageCard] = []
    for job in jobs:
        retrieval = retrieval_by_job.get(job.usage_job_id)
        if not retrieval or not retrieval.passages:
            continue
        passage = retrieval.passages[0]
        usage_card_id = stable_id("usage_card", job.usage_job_id, passage.source_id)
        evidence_span = passage.text[:240].strip()
        claim = EvidenceClaim(
            claim_id=stable_id("claim", usage_card_id, passage.source_id),
            usage_card_id=usage_card_id,
            source_id=passage.source_id,
            claim=f"{job.canonical_name} is relevant to {job.subject} reasoning.",
            evidence_span=evidence_span,
            critical=True,
            status="UNSUPPORTED",
        )
        cards.append(
            UsageCard(
                usage_card_id=usage_card_id,
                usage_job_id=job.usage_job_id,
                concept_id=job.concept_id,
                status="provisional",
                rule=f"Use evidence about {job.canonical_name} when the seed project requires {job.reasoning_operator}.",
                trigger_conditions=[f"Question context matches {job.canonical_name}.", *job.seed_project_ids[:1]],
                decision_procedure=[
                    "Read the question and options.",
                    "Match the requested relation to the retrieved concept evidence.",
                    "Apply the rule without using the gold answer.",
                ],
                failure_boundaries=["Do not use this card when retrieved evidence is missing or conflicting."],
                confusions=[f"Adjacent concepts in {job.subject} may look similar."],
                evidence_claims=[claim],
            )
        )
    return cards[: len(jobs) * config.usage_card.max_cards_per_usage_job]


async def _extract_usage_cards_model(
    jobs: list[UsageJob],
    retrieval_results: list[RetrievalResult],
    config: SmokeTestConfig,
    *,
    client: SmokeModelClient,
    errors_path: Path | None,
) -> list[UsageCard]:
    retrieval_by_job = {result.usage_job_id: result for result in retrieval_results}
    tasks = [
        _extract_one_usage_cards(job, retrieval_by_job.get(job.usage_job_id), config, client=client, errors_path=errors_path)
        for job in jobs
    ]
    nested = await asyncio.gather(*tasks)
    return [card for cards in nested for card in cards]


async def _extract_one_usage_cards(
    job: UsageJob,
    retrieval: RetrievalResult | None,
    config: SmokeTestConfig,
    *,
    client: SmokeModelClient,
    errors_path: Path | None,
) -> list[UsageCard]:
    if not retrieval or not retrieval.passages:
        if errors_path is not None:
            append_error(errors_path, stage="usage_card", item_id=job.usage_job_id, error="no_retrieved_passages")
        return []
    messages = [
        {
            "role": "system",
            "content": (
                "You create evidence-grounded Usage Cards for a smoke test. Return only valid JSON. "
                "Use only the supplied passages. Do not use answer letters, gold answers, or source-specific option wording. "
                "Every evidence_span must be copied exactly from the passage text."
            ),
        },
        {"role": "user", "content": _usage_card_prompt(job, retrieval, config)},
    ]
    value, _result, error = await client.complete_json(
        messages,
        namespace=f"usage_card.{job.usage_job_id}",
        validate=lambda payload: _cards_from_payload(payload, job, retrieval, config),
    )
    if isinstance(value, list):
        return value
    if errors_path is not None:
        append_error(errors_path, stage="usage_card", item_id=job.usage_job_id, error=error or "usage_card_extraction_failed")
    return []


def _usage_card_prompt(job: UsageJob, retrieval: RetrievalResult, config: SmokeTestConfig) -> str:
    passages = []
    for passage in retrieval.passages[: config.retrieval.final_passages_per_job]:
        text = passage.text.replace("\n", " ").strip()
        passages.append(
            f"source_id: {passage.source_id}\n"
            f"title: {passage.title or ''}\n"
            f"section: {passage.section or ''}\n"
            f"text: {text[:1400]}"
        )
    passage_block = "\n\n---\n\n".join(passages)
    queries = "\n".join(f"- {query}" for query in job.retrieval_queries)
    return f"""Create at most {config.usage_card.max_cards_per_usage_job} reusable Usage Cards for this usage job.

Return this JSON object:
{{
  "cards": [
    {{
      "rule": "reusable rule grounded in evidence",
      "trigger_conditions": ["when to use this card"],
      "decision_procedure": ["step 1", "step 2"],
      "failure_boundaries": ["when not to use this card"],
      "confusions": ["nearby confusion"],
      "evidence_claims": [
        {{
          "source_id": "one supplied source_id",
          "claim": "atomic evidence-grounded claim",
          "evidence_span": "exact substring copied from that source text",
          "critical": true
        }}
      ]
    }}
  ]
}}

Usage job:
- subject: {job.subject}
- canonical_name: {job.canonical_name}
- usage_type: {job.usage_type}
- reasoning_operator: {job.reasoning_operator}
- retrieval_queries:
{queries}

Passages:
{passage_block}
"""


def _cards_from_payload(payload: dict[str, Any], job: UsageJob, retrieval: RetrievalResult, config: SmokeTestConfig) -> list[UsageCard]:
    raw_cards = payload.get("cards")
    if not isinstance(raw_cards, list):
        raise ValueError("cards must be a list")
    source_ids = {passage.source_id for passage in retrieval.passages}
    cards: list[UsageCard] = []
    for card_idx, raw_card in enumerate(raw_cards[: config.usage_card.max_cards_per_usage_job]):
        if not isinstance(raw_card, dict):
            continue
        usage_card_id = stable_id("usage_card", job.usage_job_id, card_idx, raw_card.get("rule", ""))
        claims: list[EvidenceClaim] = []
        raw_claims = raw_card.get("evidence_claims")
        if not isinstance(raw_claims, list):
            raw_claims = []
        for claim_idx, raw_claim in enumerate(raw_claims[:6]):
            if not isinstance(raw_claim, dict):
                continue
            source_id = str(raw_claim.get("source_id", "")).strip()
            if source_id not in source_ids:
                continue
            evidence_span = str(raw_claim.get("evidence_span", "")).strip()
            claim_text = str(raw_claim.get("claim", "")).strip()
            if not evidence_span or not claim_text:
                continue
            claims.append(
                EvidenceClaim(
                    claim_id=stable_id("claim", usage_card_id, source_id, claim_idx),
                    usage_card_id=usage_card_id,
                    source_id=source_id,
                    claim=claim_text[:500],
                    evidence_span=evidence_span[:500],
                    critical=bool(raw_claim.get("critical", True)),
                    status="UNSUPPORTED",
                )
            )
        if not claims:
            continue
        cards.append(
            UsageCard(
                usage_card_id=usage_card_id,
                usage_job_id=job.usage_job_id,
                concept_id=job.concept_id,
                status="provisional",
                rule=_short(raw_card.get("rule"), default=f"Apply evidence for {job.canonical_name}.", limit=800),
                trigger_conditions=_list_str(raw_card.get("trigger_conditions"))[:8] or [f"Problem requires {job.canonical_name}."],
                decision_procedure=_list_str(raw_card.get("decision_procedure"))[:8] or ["Match the problem to the evidence-grounded rule."],
                failure_boundaries=_list_str(raw_card.get("failure_boundaries"))[:8] or ["Do not use when evidence is absent."],
                confusions=_list_str(raw_card.get("confusions"))[:8],
                evidence_claims=claims,
            )
        )
    if not cards:
        raise ValueError("no valid usage cards")
    return cards


def _list_str(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _short(value: Any, *, default: str, limit: int) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return default
    return text[:limit].strip()
