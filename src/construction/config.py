"""Model settings for the single English Wikipag construction pipeline."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MODEL = "qwen3.5-9b"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default=DEFAULT_MODEL, min_length=1)
    endpoint: Literal["chat/completions", "responses"] = "chat/completions"
    reasoning_effort: str | None = "none"
    max_completion_tokens: int = Field(default=6144, gt=0)
    concurrency: int = Field(default=8, ge=1, le=32)
    max_retries: int = Field(default=2, ge=0, le=2)
    cache_enabled: bool = True
