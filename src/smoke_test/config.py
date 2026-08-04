from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    output_dir: Path = Path("runs/smoke_001/wiki_clean")
    seed: int = 42
    resume: bool = True


class WikipagConfig(BaseModel):
    service_url: str = "http://127.0.0.1:8897"
    top_k_per_query: int = Field(default=8, ge=1, le=100)
    retrieval_timeout_s: float = 120.0


class ConstructionConfig(BaseModel):
    target_active_concepts: int = Field(default=20, ge=1)
    max_articles: int = Field(default=30, ge=1)
    max_passages: int = Field(default=24, ge=1)
    max_seed_queries: int = Field(default=8, ge=1)
    max_extraction_passages: int = Field(default=8, ge=1)
    max_grounding_candidates: int = Field(default=30, ge=1)
    grounding_top_k: int = Field(default=3, ge=1)
    max_usage_concepts: int = Field(default=5, ge=1)
    max_usage_jobs: int = Field(default=10, ge=1)
    usage_retrieval_top_k: int = Field(default=8, ge=1)
    final_materials_per_usage_job: int = Field(default=4, ge=1)


class ModelConfig(BaseModel):
    name: str = "gpt-5.4"
    endpoint: str = "chat/completions"
    reasoning_effort: str = "low"
    max_completion_tokens: int = 2048
    concurrency: int = Field(default=4, ge=1, le=32)
    max_retries: int = Field(default=2, ge=0, le=2)
    max_total_calls: int = Field(default=800, ge=0, le=800)
    cache_enabled: bool = True


class WikiCleanConfig(BaseModel):
    run: RunConfig = RunConfig()
    wikipag: WikipagConfig = WikipagConfig()
    construction: ConstructionConfig = ConstructionConfig()
    model: ModelConfig = ModelConfig()


def load_config(path: str | Path) -> WikiCleanConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return WikiCleanConfig.model_validate(raw)


def dump_config(config: WikiCleanConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")
