from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


Answer = Literal["A", "B", "C", "D"]
ClaimStatus = Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONFLICTED"]
CardStatus = Literal["active", "provisional", "rejected"]


class ProjectRecord(BaseModel):
    project_id: str
    sample_id: str
    language: str
    subject: str
    subject_category: str
    question: str
    options: dict[str, str]
    answer: Answer
    source_dataset_split: str
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_options(self) -> "ProjectRecord":
        missing = {"A", "B", "C", "D"} - set(self.options)
        if missing:
            raise ValueError(f"missing options: {sorted(missing)}")
        return self


class UsagePattern(BaseModel):
    usage_id: str
    usage_type: str
    reasoning_operator: str
    decision_rule: str
    trigger_conditions: list[str]
    confusions: list[str]
    generation_slots: list[str]
    retrieval_queries: list[str] = Field(max_length=3)


class ConceptSignature(BaseModel):
    concept_id: str
    project_id: str
    canonical_name: str
    relation: Literal["SAME", "RELATED", "CONFUSABLE", "DISTINCT"] = "DISTINCT"
    usage_patterns: list[UsagePattern] = Field(max_length=3)


class ProjectSignature(BaseModel):
    project_id: str
    subject: str
    concepts: list[ConceptSignature] = Field(max_length=2)
    model: str
    dry_run: bool


class ConceptRecord(BaseModel):
    concept_id: str
    canonical_name: str
    subject: str
    project_ids: list[str]
    relation: str = "DISTINCT"


class UsageJob(BaseModel):
    usage_job_id: str
    subject: str
    concept_id: str
    canonical_name: str
    usage_type: str
    reasoning_operator: str
    retrieval_queries: list[str] = Field(max_length=3)
    seed_project_ids: list[str]


class RetrievedPassage(BaseModel):
    source_id: str
    usage_job_id: str
    query: str
    rank: int
    score: float
    title: str | None = None
    section: str | None = None
    passage_id: str
    text: str


class RetrievalResult(BaseModel):
    usage_job_id: str
    passages: list[RetrievedPassage] = Field(max_length=5)


class EvidenceClaim(BaseModel):
    claim_id: str
    usage_card_id: str
    source_id: str
    claim: str
    evidence_span: str
    critical: bool = True
    status: ClaimStatus = "UNSUPPORTED"


class UsageCard(BaseModel):
    usage_card_id: str
    usage_job_id: str
    concept_id: str
    status: CardStatus = "provisional"
    rule: str
    trigger_conditions: list[str]
    decision_procedure: list[str]
    failure_boundaries: list[str]
    confusions: list[str]
    evidence_claims: list[EvidenceClaim]


class ReplayResult(BaseModel):
    project_id: str
    usage_card_ids: list[str]
    without_card_answer: Answer
    with_card_answer: Answer
    gold_answer: Answer
    without_card_correct: bool
    with_card_correct: bool
    dry_run: bool
