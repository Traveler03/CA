from __future__ import annotations

import json
from pathlib import Path

import httpx

from src.smoke_test.config import SmokeTestConfig
from src.smoke_test.schemas import RetrievedPassage, RetrievalResult, UsageJob
from src.smoke_test.stage_utils import stable_id


def retrieve_wikipag(jobs: list[UsageJob], config: SmokeTestConfig, *, cache_dir: Path) -> list[RetrievalResult]:
    if not config.retrieval.read_only:
        raise ValueError("Wikipag retrieval must be read-only in smoke_001")
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[RetrievalResult] = []
    endpoint = config.retrieval.service_url.rstrip("/") + "/search_batch"

    flattened: list[tuple[UsageJob, str]] = []
    for job in jobs:
        for query in job.retrieval_queries[:3]:
            flattened.append((job, query))

    by_job: dict[str, list[RetrievedPassage]] = {job.usage_job_id: [] for job in jobs}
    with httpx.Client(timeout=config.retrieval.timeout_s) as client:
        for start in range(0, len(flattened), 32):
            batch = flattened[start : start + 32]
            queries = [query for _job, query in batch]
            response = client.post(endpoint, json={"queries": queries, "top_k": config.retrieval.top_k_per_query})
            response.raise_for_status()
            payload = response.json()
            for (job, query), item in zip(batch, payload.get("results", [])):
                cache_path = cache_dir / f"{stable_id('retrieval', job.usage_job_id, query)}.json"
                cache_path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                for hit in item.get("results", []):
                    passage_id = str(hit.get("passage_id"))
                    source_id = stable_id("source", job.usage_job_id, passage_id)
                    passage = RetrievedPassage(
                        source_id=source_id,
                        usage_job_id=job.usage_job_id,
                        query=query,
                        rank=int(hit.get("rank", 0)),
                        score=float(hit.get("score", 0.0)),
                        title=hit.get("title"),
                        section=hit.get("section"),
                        passage_id=passage_id,
                        text=str(hit.get("text", "")),
                    )
                    by_job[job.usage_job_id].append(passage)

    for job in jobs:
        seen: set[str] = set()
        deduped: list[RetrievedPassage] = []
        for passage in sorted(by_job[job.usage_job_id], key=lambda row: (-row.score, row.rank, row.passage_id)):
            if passage.passage_id in seen:
                continue
            seen.add(passage.passage_id)
            deduped.append(passage)
            if len(deduped) >= config.retrieval.final_passages_per_job:
                break
        results.append(RetrievalResult(usage_job_id=job.usage_job_id, passages=deduped))
    return results
