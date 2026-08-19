from __future__ import annotations

from collections import defaultdict

from src.smoke_test.schemas import ConceptRecord, ProjectSignature, UsageJob
from src.smoke_test.stage_utils import stable_id


def build_concepts(signatures: list[ProjectSignature]) -> list[ConceptRecord]:
    grouped: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    concept_id_by_key: dict[tuple[str, str, str], str] = {}
    relation_by_key: dict[tuple[str, str, str], str] = {}
    for signature in signatures:
        for concept in signature.concepts:
            key = (signature.subject, concept.canonical_name, concept.relation)
            grouped[key].add(signature.project_id)
            concept_id_by_key[key] = concept.concept_id
            relation_by_key[key] = concept.relation
    records = [
        ConceptRecord(
            concept_id=concept_id_by_key[key],
            canonical_name=key[1],
            subject=key[0],
            project_ids=sorted(project_ids),
            relation=relation_by_key[key],
        )
        for key, project_ids in grouped.items()
    ]
    records.sort(key=lambda row: (row.subject, row.canonical_name, row.concept_id))
    return records


def build_usage_jobs(signatures: list[ProjectSignature]) -> list[UsageJob]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    for signature in signatures:
        for concept in signature.concepts:
            for pattern in concept.usage_patterns:
                key = (
                    signature.subject,
                    concept.concept_id,
                    concept.canonical_name,
                    pattern.usage_type,
                    pattern.reasoning_operator,
                )
                item = grouped.setdefault(
                    key,
                    {
                        "queries": [],
                        "projects": [],
                    },
                )
                item["projects"].append(signature.project_id)  # type: ignore[index]
                for query in pattern.retrieval_queries:
                    if query not in item["queries"]:  # type: ignore[operator]
                        item["queries"].append(query)  # type: ignore[index]

    jobs: list[UsageJob] = []
    for key, item in grouped.items():
        subject, concept_id, canonical_name, usage_type, reasoning_operator = key
        usage_job_id = stable_id("usage_job", *key)
        jobs.append(
            UsageJob(
                usage_job_id=usage_job_id,
                subject=subject,
                concept_id=concept_id,
                canonical_name=canonical_name,
                usage_type=usage_type,
                reasoning_operator=reasoning_operator,
                retrieval_queries=list(item["queries"])[:3],  # type: ignore[index]
                seed_project_ids=sorted(set(item["projects"])),  # type: ignore[arg-type]
            )
        )
    jobs.sort(key=lambda row: (row.subject, row.canonical_name, row.usage_job_id))
    return jobs
