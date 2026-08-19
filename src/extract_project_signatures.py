from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from src.smoke_test.io import append_error
from src.smoke_test.model_client import SmokeModelClient
from src.smoke_test.config import SmokeTestConfig
from src.smoke_test.schemas import ConceptSignature, ProjectRecord, ProjectSignature, UsagePattern
from src.smoke_test.stage_utils import keyword_candidates, normalize_name, stable_id


def extract_project_signatures(
    projects: list[ProjectRecord],
    config: SmokeTestConfig,
    *,
    dry_run: bool,
    client: SmokeModelClient | None = None,
    errors_path: Path | None = None,
) -> list[ProjectSignature]:
    if not dry_run:
        if client is None:
            raise ValueError("model-backed signature extraction requires SmokeModelClient")
        return asyncio.run(_extract_project_signatures_model(projects, config, client=client, errors_path=errors_path))
    signatures: list[ProjectSignature] = []
    for project in projects:
        text = " ".join([project.subject, project.question, *project.options.values()])
        names = keyword_candidates(text, limit=config.signature.max_concepts_per_project)
        if not names:
            names = [normalize_name(project.subject)]
        concepts = []
        for name in names[: config.signature.max_concepts_per_project]:
            canonical = normalize_name(name)
            concept_id = stable_id("concept", project.subject, canonical)
            usage_id = stable_id("usage", project.project_id, canonical, "dry_run")
            retrieval_queries = [
                f"{canonical} {project.subject}",
                f"{canonical} decision rule",
                f"{canonical} common confusion",
            ][: config.signature.max_retrieval_queries]
            pattern = UsagePattern(
                usage_id=usage_id,
                usage_type="dry_run_reasoning_usage",
                reasoning_operator="identify_and_apply_rule",
                decision_rule=f"Determine whether the question requires the {canonical} concept.",
                trigger_conditions=[f"The problem mentions or implies {canonical}."],
                confusions=[f"Confusing {canonical} with adjacent {project.subject} concepts."],
                generation_slots=["given_conditions", "target_quantity_or_relation", "answer_options"],
                retrieval_queries=retrieval_queries,
            )
            concepts.append(
                ConceptSignature(
                    concept_id=concept_id,
                    project_id=project.project_id,
                    canonical_name=canonical,
                    relation="DISTINCT",
                    usage_patterns=[pattern],
                )
            )
        signatures.append(
            ProjectSignature(
                project_id=project.project_id,
                subject=project.subject,
                concepts=concepts,
                model=config.model.name,
                dry_run=True,
            )
        )
    return signatures


async def _extract_project_signatures_model(
    projects: list[ProjectRecord],
    config: SmokeTestConfig,
    *,
    client: SmokeModelClient,
    errors_path: Path | None,
) -> list[ProjectSignature]:
    tasks = [_extract_one_project_signature(project, config, client=client, errors_path=errors_path) for project in projects]
    rows = await asyncio.gather(*tasks)
    return [row for row in rows if row is not None]


async def _extract_one_project_signature(
    project: ProjectRecord,
    config: SmokeTestConfig,
    *,
    client: SmokeModelClient,
    errors_path: Path | None,
) -> ProjectSignature | None:
    messages = [
        {
            "role": "system",
            "content": (
                "You extract reusable concept-usage signatures for a small Global-MMLU smoke test. "
                "Return only valid JSON. Do not reveal hidden reasoning. Do not use or infer the gold answer. "
                "Do not copy option letters into reusable rules."
            ),
        },
        {
            "role": "user",
            "content": _signature_prompt(project, config),
        },
    ]
    value, _result, error = await client.complete_json(
        messages,
        namespace=f"signature.{project.project_id}",
        validate=lambda payload: _signature_from_payload(payload, project, config),
    )
    if isinstance(value, ProjectSignature):
        return value
    if errors_path is not None:
        append_error(
            errors_path,
            stage="project_signature",
            item_id=project.project_id,
            error=error or "signature_extraction_failed",
            payload={"project_id": project.project_id},
        )
    return None


def _signature_prompt(project: ProjectRecord, config: SmokeTestConfig) -> str:
    options = "\n".join(f"{label}. {text}" for label, text in sorted(project.options.items()))
    return f"""Extract at most {config.signature.max_concepts_per_project} core concepts from this multiple-choice project.

For each concept, output at most {config.signature.max_usage_patterns_per_concept} usage patterns.
Each usage pattern must include at most {config.signature.max_retrieval_queries} retrieval_queries.

Return this JSON object exactly:
{{
  "concepts": [
    {{
      "canonical_name": "short lowercase reusable concept name",
      "relation": "SAME|RELATED|CONFUSABLE|DISTINCT",
      "usage_patterns": [
        {{
          "usage_type": "definition|classification|causal_reasoning|quantitative_rule|diagnosis|legal_rule|historical_reasoning|other",
          "reasoning_operator": "short operator name",
          "decision_rule": "reusable rule without answer letters or source-specific wording",
          "trigger_conditions": ["observable condition"],
          "confusions": ["nearby misconception"],
          "generation_slots": ["variable slot"],
          "retrieval_queries": ["wikipedia search query"]
        }}
      ]
    }}
  ]
}}

Subject: {project.subject}
Question:
{project.question}

Options:
{options}
"""


def _signature_from_payload(payload: dict[str, Any], project: ProjectRecord, config: SmokeTestConfig) -> ProjectSignature:
    raw_concepts = payload.get("concepts")
    if not isinstance(raw_concepts, list) or not raw_concepts:
        raise ValueError("concepts must be a non-empty list")

    concepts: list[ConceptSignature] = []
    seen_names: set[str] = set()
    for raw_concept in raw_concepts[: config.signature.max_concepts_per_project]:
        if not isinstance(raw_concept, dict):
            continue
        canonical = normalize_name(str(raw_concept.get("canonical_name", "")))
        if not canonical or canonical in seen_names:
            continue
        seen_names.add(canonical)
        relation = str(raw_concept.get("relation", "DISTINCT")).upper()
        if relation not in {"SAME", "RELATED", "CONFUSABLE", "DISTINCT"}:
            relation = "DISTINCT"
        concept_id = stable_id("concept", project.subject, canonical)
        patterns: list[UsagePattern] = []
        raw_patterns = raw_concept.get("usage_patterns")
        if not isinstance(raw_patterns, list):
            raw_patterns = []
        for idx, raw_pattern in enumerate(raw_patterns[: config.signature.max_usage_patterns_per_concept]):
            if not isinstance(raw_pattern, dict):
                continue
            queries = _list_str(raw_pattern.get("retrieval_queries"))[: config.signature.max_retrieval_queries]
            if not queries:
                queries = [f"{canonical} {project.subject}"]
            pattern = UsagePattern(
                usage_id=stable_id("usage", project.project_id, canonical, idx),
                usage_type=_short(raw_pattern.get("usage_type"), default="other"),
                reasoning_operator=_short(raw_pattern.get("reasoning_operator"), default="apply_rule"),
                decision_rule=_short(raw_pattern.get("decision_rule"), default=f"Apply the reusable rule for {canonical}.", limit=500),
                trigger_conditions=_list_str(raw_pattern.get("trigger_conditions"))[:6] or [f"Problem requires {canonical}."],
                confusions=_list_str(raw_pattern.get("confusions"))[:6],
                generation_slots=_list_str(raw_pattern.get("generation_slots"))[:8],
                retrieval_queries=queries,
            )
            patterns.append(pattern)
        if not patterns:
            patterns.append(
                UsagePattern(
                    usage_id=stable_id("usage", project.project_id, canonical, "fallback"),
                    usage_type="other",
                    reasoning_operator="apply_rule",
                    decision_rule=f"Apply the reusable rule for {canonical}.",
                    trigger_conditions=[f"Problem requires {canonical}."],
                    confusions=[],
                    generation_slots=[],
                    retrieval_queries=[f"{canonical} {project.subject}"][: config.signature.max_retrieval_queries],
                )
            )
        concepts.append(
            ConceptSignature(
                concept_id=concept_id,
                project_id=project.project_id,
                canonical_name=canonical,
                relation=relation,  # type: ignore[arg-type]
                usage_patterns=patterns,
            )
        )

    if not concepts:
        raise ValueError("no valid concepts extracted")
    return ProjectSignature(project_id=project.project_id, subject=project.subject, concepts=concepts, model=config.model.name, dry_run=False)


def _list_str(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _short(value: Any, *, default: str, limit: int = 120) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return default
    return text[:limit].strip()
