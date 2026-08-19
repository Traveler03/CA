from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from src.smoke_test.io import append_error
from src.smoke_test.model_client import SmokeModelClient
from src.smoke_test.schemas import Answer, ProjectRecord, ReplayResult, UsageCard, UsageJob


def replay_seed_projects(
    projects: list[ProjectRecord],
    jobs: list[UsageJob],
    cards: list[UsageCard],
    *,
    dry_run: bool,
    client: SmokeModelClient | None = None,
    errors_path: Path | None = None,
) -> list[ReplayResult]:
    if not dry_run:
        if client is None:
            raise ValueError("model-backed seed replay requires SmokeModelClient")
        return asyncio.run(_replay_seed_projects_model(projects, jobs, cards, client=client, errors_path=errors_path))
    active_by_job = {card.usage_job_id: card for card in cards if card.status == "active"}
    cards_by_project: dict[str, list[str]] = {project.project_id: [] for project in projects}
    for job in jobs:
        card = active_by_job.get(job.usage_job_id)
        if not card:
            continue
        for project_id in job.seed_project_ids:
            cards_by_project.setdefault(project_id, []).append(card.usage_card_id)

    results: list[ReplayResult] = []
    for project in projects:
        without_answer: Answer = "A"
        with_answer: Answer = "A"
        results.append(
            ReplayResult(
                project_id=project.project_id,
                usage_card_ids=cards_by_project.get(project.project_id, []),
                without_card_answer=without_answer,
                with_card_answer=with_answer,
                gold_answer=project.answer,
                without_card_correct=without_answer == project.answer,
                with_card_correct=with_answer == project.answer,
                dry_run=True,
            )
        )
    return results


async def _replay_seed_projects_model(
    projects: list[ProjectRecord],
    jobs: list[UsageJob],
    cards: list[UsageCard],
    *,
    client: SmokeModelClient,
    errors_path: Path | None,
) -> list[ReplayResult]:
    cards_by_project = _cards_by_project(jobs, cards)
    tasks = [_replay_one_project(project, cards_by_project.get(project.project_id, []), client=client, errors_path=errors_path) for project in projects]
    rows = await asyncio.gather(*tasks)
    return rows


async def _replay_one_project(
    project: ProjectRecord,
    cards: list[UsageCard],
    *,
    client: SmokeModelClient,
    errors_path: Path | None,
) -> ReplayResult:
    without_answer, without_error = await _answer_project(project, [], client=client, namespace=f"replay.without.{project.project_id}")
    with_answer, with_error = await _answer_project(project, cards, client=client, namespace=f"replay.with.{project.project_id}")
    if errors_path is not None:
        if without_error:
            append_error(errors_path, stage="seed_replay_without_card", item_id=project.project_id, error=without_error)
        if with_error:
            append_error(errors_path, stage="seed_replay_with_card", item_id=project.project_id, error=with_error)
    return ReplayResult(
        project_id=project.project_id,
        usage_card_ids=[card.usage_card_id for card in cards],
        without_card_answer=without_answer,
        with_card_answer=with_answer,
        gold_answer=project.answer,
        without_card_correct=without_answer == project.answer,
        with_card_correct=with_answer == project.answer,
        dry_run=False,
    )


async def _answer_project(
    project: ProjectRecord,
    cards: list[UsageCard],
    *,
    client: SmokeModelClient,
    namespace: str,
) -> tuple[Answer, str | None]:
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the multiple-choice question. Return only JSON. Do not reveal hidden reasoning. "
                "Use any supplied Usage Cards as reusable guidance, not as source-specific answer keys."
            ),
        },
        {"role": "user", "content": _replay_prompt(project, cards)},
    ]
    value, _result, error = await client.complete_json(messages, namespace=namespace, validate=_answer_from_payload)
    if value in {"A", "B", "C", "D"}:
        return value, None  # type: ignore[return-value]
    return "A", error or "answer_parse_failed"


def _cards_by_project(jobs: list[UsageJob], cards: list[UsageCard]) -> dict[str, list[UsageCard]]:
    active_by_job = {card.usage_job_id: card for card in cards if card.status == "active"}
    cards_by_project: dict[str, list[UsageCard]] = {}
    for job in jobs:
        card = active_by_job.get(job.usage_job_id)
        if not card:
            continue
        for project_id in job.seed_project_ids:
            cards_by_project.setdefault(project_id, []).append(card)
    for project_id, project_cards in cards_by_project.items():
        seen: set[str] = set()
        deduped: list[UsageCard] = []
        for card in project_cards:
            if card.usage_card_id in seen:
                continue
            seen.add(card.usage_card_id)
            deduped.append(card)
        cards_by_project[project_id] = deduped[:4]
    return cards_by_project


def _replay_prompt(project: ProjectRecord, cards: list[UsageCard]) -> str:
    options = "\n".join(f"{label}. {text}" for label, text in sorted(project.options.items()))
    card_block = "No usage cards supplied."
    if cards:
        snippets = []
        for card in cards[:4]:
            snippets.append(
                "Usage Card\n"
                f"- rule: {card.rule}\n"
                f"- trigger_conditions: {'; '.join(card.trigger_conditions[:4])}\n"
                f"- decision_procedure: {'; '.join(card.decision_procedure[:4])}\n"
                f"- failure_boundaries: {'; '.join(card.failure_boundaries[:4])}"
            )
        card_block = "\n\n".join(snippets)
    return f"""Return exactly this JSON shape:
{{"answer":"A|B|C|D"}}

Subject: {project.subject}

Question:
{project.question}

Options:
{options}

Usage cards:
{card_block}
"""


def _answer_from_payload(payload: dict[str, Any]) -> Answer:
    answer = str(payload.get("answer", "")).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError("answer must be one of A/B/C/D")
    return answer  # type: ignore[return-value]
