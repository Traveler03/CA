from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class RunConfig(BaseModel):
    name: str = "smoke_001"
    seed: int = 42
    output_dir: Path = Path("runs/smoke_001")
    resume: bool = True
    dry_run: bool = False


class DatasetConfig(BaseModel):
    path: Path = Path("data/processed/global_mmlu/en.jsonl")
    language: str = "en"
    total_projects: int = Field(default=100, le=100)
    projects_per_subject: int = Field(default=20, le=20)
    subjects: list[str]
    exclude_time_sensitive: bool = True
    exclude_culturally_sensitive: bool = True

    @model_validator(mode="after")
    def _validate_scope(self) -> "DatasetConfig":
        if len(self.subjects) != 5:
            raise ValueError("smoke_001 must use exactly 5 subjects")
        if self.total_projects > 100:
            raise ValueError("smoke_001 cannot exceed 100 projects")
        if self.projects_per_subject > 20:
            raise ValueError("smoke_001 cannot exceed 20 projects per subject")
        return self


class SignatureConfig(BaseModel):
    max_concepts_per_project: int = Field(default=2, ge=1, le=2)
    max_usage_patterns_per_concept: int = Field(default=3, ge=1, le=3)
    max_retrieval_queries: int = Field(default=3, ge=1, le=3)


class RetrievalConfig(BaseModel):
    service_url: str = "http://127.0.0.1:8897"
    top_k_per_query: int = Field(default=20, ge=1, le=20)
    final_passages_per_job: int = Field(default=5, ge=1, le=5)
    read_only: bool = True
    timeout_s: float = 120.0


class UsageCardConfig(BaseModel):
    max_cards_per_usage_job: int = Field(default=2, ge=1, le=2)
    require_evidence_span: bool = True
    require_all_critical_claims_supported: bool = True


class ModelConfig(BaseModel):
    name: str = "gpt-5.5"
    endpoint: str = "chat/completions"
    reasoning_effort: str = "low"
    max_completion_tokens: int = 2048
    concurrency: int = Field(default=4, ge=1, le=32)
    max_retries: int = Field(default=2, ge=0, le=2)
    max_total_calls: int = Field(default=800, ge=0, le=800)
    cache_enabled: bool = True


class CanaryConfig(BaseModel):
    project_count: int = Field(default=20, ge=1, le=20)
    stop_on_error_rate: float = Field(default=0.20, ge=0.0, le=1.0)


class ReplayConfig(BaseModel):
    enabled: bool = True
    compare_without_card: bool = True


class WriteConfig(BaseModel):
    production_registry: bool = False
    production_vector_index: bool = False

    @model_validator(mode="after")
    def _validate_read_only(self) -> "WriteConfig":
        if self.production_registry or self.production_vector_index:
            raise ValueError("smoke_001 must not write production registry or vector index")
        return self


class SmokeTestConfig(BaseModel):
    run: RunConfig
    dataset: DatasetConfig
    signature: SignatureConfig
    retrieval: RetrievalConfig
    usage_card: UsageCardConfig
    model: ModelConfig
    canary: CanaryConfig
    replay: ReplayConfig
    write: WriteConfig

    @model_validator(mode="after")
    def _validate_budget(self) -> "SmokeTestConfig":
        if self.model.max_total_calls > 800:
            raise ValueError("model-call budget cannot exceed 800")
        if self.dataset.total_projects > 100:
            raise ValueError("project count cannot exceed 100")
        return self


def load_config(path: str | Path) -> SmokeTestConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SmokeTestConfig.model_validate(raw)


def dump_config(config: SmokeTestConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")
