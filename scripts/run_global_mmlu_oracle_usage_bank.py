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
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.smoke_test.config import ModelConfig
from src.smoke_test.io import read_jsonl, write_jsonl
from src.smoke_test.model_client import SmokeModelClient
from src.smoke_test.stage_utils import normalize_name, stable_id


DEFAULT_INPUT = Path("data/processed/global_mmlu/en.jsonl")
DEFAULT_OUTPUT = Path("artifacts/ca_mem/global_mmlu_oracle_cards_en_gpt54_v1")
VALID_CLAIM_FIELDS = {
    "description",
    "trigger",
    "decision",
    "pitfall",
}


class OracleEvidenceClaim(BaseModel):
    field: str
    claim: str
    source_ids: list[str] = Field(default_factory=list)
    support_type: str = "derived"

    @model_validator(mode="after")
    def _normalize(self) -> "OracleEvidenceClaim":
        if self.field not in VALID_CLAIM_FIELDS:
            self.field = "decision"
        if self.support_type not in {"direct", "derived"}:
            self.support_type = "derived"
        self.source_ids = [source_id for source_id in self.source_ids if source_id]
        return self


class OracleCardDraft(BaseModel):
    concept: str
    description: str
    trigger: list[str] = Field(default_factory=list)
    decision: list[str] = Field(default_factory=list)
    pitfall: list[str] = Field(default_factory=list)
    evidence_claims: list[OracleEvidenceClaim] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_card(self) -> "OracleCardDraft":
        if not self.concept.strip():
            raise ValueError("concept is required")
        if not self.description.strip():
            raise ValueError("description is required")
        if not self.trigger:
            raise ValueError("trigger is required")
        if not self.decision:
            raise ValueError("decision is required")
        if not self.pitfall:
            raise ValueError("pitfall is required")
        return self


class OracleExtractionPayload(BaseModel):
    cards: list[OracleCardDraft] = Field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build English question-derived oracle Usage Cards. "
            "These cards use benchmark questions and gold answer text, so they are diagnostic-only."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-dataset", default="global_mmlu")
    parser.add_argument("--dataset-label")
    parser.add_argument("--bank-component")
    parser.add_argument("--row-key-prefix")
    parser.add_argument("--namespace-salt", default="")
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--subjects", nargs="+", default=["all"])
    parser.add_argument("--source-splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--exclude-global-mmlu-overlap", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0, help="0 means no global row cap.")
    parser.add_argument("--max-rows-per-subject", type=int, default=1000, help="0 means no per-subject cap.")
    parser.add_argument("--cards-per-question", type=int, default=1)
    parser.add_argument("--backend", choices=["api", "local-vllm"], default="api")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-completion-tokens", type=int, default=1600)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--model-timeout-s", type=float, default=180.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-model-calls", type=int, default=100000)
    parser.add_argument("--bank-version", default="global-mmlu-en-oracle-gpt54-v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Create deterministic heuristic cards without model calls.")
    parser.add_argument("--local-model-dir", type=Path, default=Path("data/external/models/Qwen3-8B"))
    parser.add_argument("--local-max-model-len", type=int, default=32768)
    parser.add_argument("--local-gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--local-batch-timeout-ms", type=int, default=10)
    parser.add_argument("--local-batch-max-tokens", type=int, default=4096)
    parser.add_argument("--write-runtime-bank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build-embedding-index", action="store_true")
    parser.add_argument("--embedding-model-dir", type=Path, default=Path("data/external/models/Qwen3-Embedding-4B"))
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-truncate-dim", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def dataset_label(args: argparse.Namespace) -> str:
    if args.dataset_label:
        return str(args.dataset_label)
    labels = {
        "global_mmlu": "Global-MMLU",
        "mmlu_prox": "MMLU-ProX",
    }
    return labels.get(str(args.source_dataset), str(args.source_dataset))


def bank_component(args: argparse.Namespace) -> str:
    return str(args.bank_component or f"{args.source_dataset}_oracle")


def row_key_prefix(args: argparse.Namespace) -> str:
    return str(args.row_key_prefix or args.source_dataset)


def construction_mode(args: argparse.Namespace) -> str:
    return f"{args.source_dataset}_english_question_answer_oracle"


def provenance_warning(args: argparse.Namespace) -> str:
    return f"Diagnostic/oracle bank derived from {dataset_label(args)} English questions and verified gold answer text."


def option_text(row: dict[str, Any], label: str | None = None) -> str:
    options = row.get("options") or {}
    answer = str(label or row.get("answer") or "").strip().upper()
    if isinstance(options, dict):
        return str(options.get(answer) or "")
    if isinstance(options, list):
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if answer in labels:
            idx = labels.index(answer)
            if idx < len(options):
                return str(options[idx])
    return str(row.get("answer_text") or "")


def format_options(options: Any) -> str:
    if isinstance(options, dict):
        return "\n".join(f"{label}. {text}" for label, text in options.items())
    if isinstance(options, list):
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "\n".join(f"{labels[idx]}. {text}" for idx, text in enumerate(options))
    return str(options or "")


def source_id_for_row(row: dict[str, Any], *, source_dataset: str) -> str:
    return stable_id(f"{source_dataset}_oracle_source", row.get("sample_id"), row.get("subject"), row.get("answer"))


def row_key(row: dict[str, Any], *, prefix: str, language: str) -> str:
    return f"{prefix}|{language}|{row.get('sample_id')}"


def select_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    wanted_subjects = set(args.subjects)
    all_subjects = "all" in wanted_subjects
    wanted_splits = set(args.source_splits or [])
    language = str(args.source_language or "en")
    source_dataset = str(args.source_dataset)
    key_prefix = row_key_prefix(args)
    per_subject = Counter()
    for row in read_jsonl(args.input_jsonl):
        if str(row.get("language") or language) != language:
            continue
        if args.exclude_global_mmlu_overlap and bool(row.get("excluded_global_mmlu_overlap")):
            continue
        subject = str(row.get("subject") or "")
        if not all_subjects and subject not in wanted_subjects:
            continue
        split = str(row.get("source_dataset_split") or "")
        if wanted_splits and split not in wanted_splits:
            continue
        if args.max_rows_per_subject and per_subject[subject] >= int(args.max_rows_per_subject):
            continue
        selected = dict(row)
        selected["_eval_key"] = row_key(row, prefix=key_prefix, language=language)
        selected["_source_id"] = source_id_for_row(row, source_dataset=source_dataset)
        selected["_gold_answer_text"] = option_text(row)
        selected["_source_dataset"] = source_dataset
        selected["_source_language"] = language
        selected["_dataset_label"] = dataset_label(args)
        selected["_bank_component"] = bank_component(args)
        selected["_construction_mode"] = construction_mode(args)
        selected["_namespace_salt"] = str(args.namespace_salt or "")
        rows.append(selected)
        per_subject[subject] += 1
        if args.max_rows and len(rows) >= int(args.max_rows):
            break
    return rows


def oracle_card_messages(row: dict[str, Any], *, cards_per_question: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You build benchmark-derived oracle Usage Cards for diagnostic research. "
                "Use the English question, options, and verified gold answer text. "
                "Do not produce a clean-evaluation card. Do not copy the answer letter into reusable rules. "
                "Do not mention 'gold answer', 'verified answer', 'benchmark', or source IDs inside reusable card fields; "
                "those provenance markers belong only in evidence_claims/source_ids. "
                "Generalize the supported answer text into exactly: concept, description, trigger, decision, pitfall. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": f"""Dataset: {row.get('_dataset_label') or row.get('_source_dataset') or 'benchmark'}
Language: {row.get('_source_language') or 'en'}
Subject: {row.get('subject') or ''}
Source split: {row.get('source_dataset_split') or ''}
Sample ID: {row.get('sample_id') or ''}

Question:
{row.get('question') or ''}

Options:
{format_options(row.get('options') or {})}

Verified gold answer text:
{row.get('_gold_answer_text') or ''}

Create up to {cards_per_question} reusable oracle card(s) grounded in this question and the verified answer text.

Requirements:
- The reusable card body must have only: concept, description, trigger, decision, pitfall.
- Keep the reusable card in English.
- Do not include option letters such as "A/B/C/D" in the rule.
- Do not put words like "gold answer", "verified answer", "benchmark", or source IDs in concept, description,
  trigger, decision, or pitfall.
- Prefer the answer text, formula, relation, or procedure over copying the full question.
- trigger = when to use this card.
- decision = how to judge/compute/apply the card.
- pitfall = common mistake, boundary, or misuse to avoid.
- evidence_claims may cite only these source IDs: ["question", "options", "gold_answer"].

Return JSON:
{{
  "cards": [
    {{
      "concept": "short reusable concept name",
      "description": "one sentence defining what the card captures",
      "trigger": ["observable condition for using this card"],
      "decision": ["step grounded in the question and supported answer text"],
      "pitfall": ["common mistake, boundary, or misuse to avoid"],
      "evidence_claims": [
        {{
          "field": "description|trigger|decision|pitfall",
          "claim": "atomic claim",
          "source_ids": ["question", "gold_answer"],
          "support_type": "direct|derived"
        }}
      ]
    }}
  ]
}}""",
        },
    ]


def validate_payload(payload: dict[str, Any]) -> OracleExtractionPayload:
    parsed = OracleExtractionPayload.model_validate(payload)
    parsed.cards = parsed.cards[:3]
    return parsed


def subject_prefix(subject: str) -> str:
    parts = [part for part in re.split(r"[_\W]+", subject.upper()) if part]
    compact = "".join(part[:4] for part in parts[:3])
    return compact[:12] or "BENCHMARK"


def card_concept_id(row: dict[str, Any], draft: OracleCardDraft, idx: int) -> str:
    slug = normalize_name(draft.concept).upper().replace(" ", "-")
    slug = re.sub(r"[^A-Z0-9-]+", "", slug).strip("-")[:80] or "CONCEPT"
    suffix = stable_id("cid", row.get("sample_id"), idx, draft.concept, length=8).split("_", 1)[1].upper()
    return f"{subject_prefix(str(row.get('subject') or ''))}-ORACLE-{slug}-{suffix}"


def normalize_claim_sources(claim: OracleEvidenceClaim, row_source_id: str) -> OracleEvidenceClaim:
    mapped = []
    for source_id in claim.source_ids:
        if source_id in {"question", "options", "gold_answer"}:
            mapped.append(f"{row_source_id}:{source_id}")
    if not mapped:
        mapped = [f"{row_source_id}:gold_answer"]
    return OracleEvidenceClaim(
        field=claim.field,
        claim=claim.claim,
        source_ids=mapped,
        support_type=claim.support_type,
    )


def scrub_reusable_text(value: str) -> str:
    text = str(value or "").strip()
    replacements = [
        (r"(?i)verified gold answer text", "supported answer text"),
        (r"(?i)verified gold answer", "supported answer"),
        (r"(?i)gold answer text", "supported answer text"),
        (r"(?i)gold_answer", "supported_answer"),
        (r"(?i)gold answer", "supported answer"),
        (r"(?i)verified answer text", "supported answer text"),
        (r"(?i)verified answer", "supported answer"),
        (r"(?i)benchmark question", "source question"),
        (r"(?i)benchmark", "source"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def scrub_reusable_list(values: list[str]) -> list[str]:
    return [scrub_reusable_text(item) for item in values if scrub_reusable_text(item)]


def build_usage_card_row(row: dict[str, Any], draft: OracleCardDraft, idx: int) -> dict[str, Any]:
    concept_id = card_concept_id(row, draft, idx)
    source_id = str(row["_source_id"])
    source_dataset = str(row.get("_source_dataset") or "global_mmlu")
    source_language = str(row.get("_source_language") or "en")
    component = str(row.get("_bank_component") or f"{source_dataset}_oracle")
    claims = [normalize_claim_sources(claim, source_id).model_dump(mode="json") for claim in draft.evidence_claims]
    if not claims:
        claims = [
            {
                "field": "decision",
                "claim": "Card decision is derived from the English question and supported answer text.",
                "source_ids": [f"{source_id}:question", f"{source_id}:gold_answer"],
                "support_type": "derived",
            }
        ]
    return {
        "usage_id": stable_id("oracle_usage", row.get("sample_id"), idx, draft.concept),
        "usage_job_id": stable_id("oracle_usage_job", row.get("sample_id"), row.get("subject")),
        "subject": row.get("subject"),
        "concept_id": concept_id,
        "concept": scrub_reusable_text(draft.concept),
        "description": scrub_reusable_text(draft.description),
        "trigger": scrub_reusable_list(draft.trigger)[:8],
        "decision": scrub_reusable_list(draft.decision)[:10],
        "pitfall": scrub_reusable_list(draft.pitfall)[:8],
        "evidence_claims": claims,
        "source_ids": [source_id],
        "source_sample_ids": [row.get("sample_id")],
        "source_dataset": source_dataset,
        "source_language": source_language,
        "source_split": row.get("source_dataset_split"),
        "source_subject_category": row.get("subject_category"),
        "bank_component": component,
        "benchmark_content_accessed": True,
        "uses_gold_answer": True,
        "can_be_used_for_clean_eval": False,
        "can_be_used_for_clean_global_mmlu_eval": False,
        "status": "active",
        "version": 1,
    }


def heuristic_payload(row: dict[str, Any], cards_per_question: int) -> OracleExtractionPayload:
    answer = str(row.get("_gold_answer_text") or "the supported answer").strip()
    subject = str(row.get("subject") or "subject")
    concept = normalize_name(answer)[:80] or f"{subject} answer pattern"
    draft = OracleCardDraft(
        concept=concept,
        description=f"Question-derived oracle concept for recognizing that {answer} is the supported answer pattern.",
        trigger=["The question asks for the same fact, relation, formula, classification, or procedure."],
        decision=[f"Map the problem to the supported answer text: {answer}."],
        pitfall=["Do not use as clean external knowledge; this is diagnostic oracle content."],
        evidence_claims=[
            OracleEvidenceClaim(
                field="decision",
                claim="The decision uses the supported answer text.",
                source_ids=["gold_answer"],
                support_type="direct",
            )
        ],
    )
    return OracleExtractionPayload(cards=[draft][:cards_per_question])


async def build_one(
    row: dict[str, Any],
    *,
    client: Any | None,
    backend: str,
    cards_per_question: int,
    dry_run: bool,
    max_completion_tokens: int,
    max_retries: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    started = time.perf_counter()
    if dry_run:
        payload = heuristic_payload(row, cards_per_question)
        result_log: dict[str, Any] = {"cached": False, "latency_s": 0.0, "usage": {}, "dry_run": True}
        error = None
    else:
        assert client is not None
        messages = oracle_card_messages(row, cards_per_question=cards_per_question)
        namespace_prefix = f"{row.get('_source_dataset') or 'global_mmlu'}_oracle_card"
        if row.get("_namespace_salt"):
            namespace_prefix = f"{namespace_prefix}.{row['_namespace_salt']}"
        namespace = f"{namespace_prefix}.{row.get('sample_id')}"
        if backend == "api":
            payload, result, error = await client.complete_json(
                messages,
                namespace=namespace,
                validate=validate_payload,
            )
            result_log = {
                "backend": backend,
                "cached": result.cached if result else None,
                "latency_s": result.latency_s if result else None,
                "usage": result.usage if result else {},
                "dry_run": False,
            }
        else:
            payload = None
            error = None
            last_raw: dict[str, Any] | None = None
            for attempt in range(1, int(max_retries) + 2):
                try:
                    raw = await client.json(
                        messages,
                        namespace=f"{namespace}.attempt_{attempt}",
                        max_tokens=max_completion_tokens,
                        temperature=0.0,
                    )
                    last_raw = raw
                    payload = validate_payload(raw)
                    error = None
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    messages = [
                        *oracle_card_messages(row, cards_per_question=cards_per_question),
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was invalid for this schema. "
                                f"Error: {error}. Return only a corrected JSON object."
                            ),
                        },
                    ]
            result_log = {
                "backend": backend,
                "cached": None,
                "latency_s": None,
                "usage": {},
                "dry_run": False,
                "last_raw_keys": sorted(last_raw.keys()) if isinstance(last_raw, dict) else [],
            }
    if not isinstance(payload, OracleExtractionPayload):
        return [], result_log | {"total_latency_s": time.perf_counter() - started}, error or "payload_validation_failed"
    cards = [build_usage_card_row(row, draft, idx) for idx, draft in enumerate(payload.cards[:cards_per_question], start=1)]
    if not cards:
        return [], result_log | {"total_latency_s": time.perf_counter() - started}, "no_cards_returned"
    return cards, result_log | {"total_latency_s": time.perf_counter() - started}, None


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def existing_eval_keys(output_dir: Path) -> set[str]:
    path = output_dir / "card_extraction_logs.jsonl"
    if not path.exists():
        return set()
    return {str(row.get("eval_key")) for row in read_jsonl(path) if row.get("ok")}


def reset_outputs(output_dir: Path) -> None:
    for file_name in [
        "source_items.jsonl",
        "usage_cards.jsonl",
        "usage_cards.raw.jsonl",
        "usage_card_claims.jsonl",
        "card_extraction_logs.jsonl",
        "errors.jsonl",
        "concept_registry.jsonl",
        "concept_index.jsonl",
        "usage_index.jsonl",
        "build_events.jsonl",
        "rejected_items.jsonl",
        "bank_manifest.json",
        "summary.json",
        "report.md",
    ]:
        path = output_dir / file_name
        if path.exists():
            path.unlink()


def build_source_item(row: dict[str, Any]) -> dict[str, Any]:
    source_id = str(row["_source_id"])
    return {
        "source_id": source_id,
        "dataset": row.get("_source_dataset") or "global_mmlu",
        "language": row.get("_source_language") or "en",
        "sample_id": row.get("sample_id"),
        "subject": row.get("subject"),
        "source_split": row.get("source_dataset_split"),
        "question": row.get("question"),
        "options": row.get("options"),
        "answer": row.get("answer"),
        "gold_answer_text": row.get("_gold_answer_text"),
        "benchmark_content_accessed": True,
        "uses_gold_answer": True,
        "source_ids": [f"{source_id}:question", f"{source_id}:options", f"{source_id}:gold_answer"],
    }


def build_claim_rows(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for card in cards:
        for idx, claim in enumerate(card.get("evidence_claims") or [], start=1):
            rows.append(
                {
                    "claim_id": stable_id("oracle_claim", card.get("usage_id"), idx, claim.get("field"), claim.get("claim")),
                    "usage_id": card.get("usage_id"),
                    "usage_job_id": card.get("usage_job_id"),
                    "concept_id": card.get("concept_id"),
                    "field": claim.get("field"),
                    "claim": claim.get("claim"),
                    "supporting_source_ids": claim.get("source_ids") or [],
                    "support_type": claim.get("support_type") or "derived",
                    "bank_component": card.get("bank_component") or f"{card.get('source_dataset') or 'global_mmlu'}_oracle",
                    "benchmark_content_accessed": True,
                    "uses_gold_answer": True,
                    "can_be_used_for_clean_eval": False,
                }
            )
    return rows


def build_concept_rows(cards: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry = []
    concept_index = []
    seen = set()
    for card in cards:
        concept_id = card["concept_id"]
        if concept_id in seen:
            continue
        seen.add(concept_id)
        registry.append(
            {
                "concept_id": concept_id,
                "subject": card.get("subject"),
                "canonical_name": card.get("concept"),
                "definition": card.get("description"),
                "status": "active",
                "source_sample_ids": card.get("source_sample_ids") or [],
                "bank_component": card.get("bank_component") or f"{card.get('source_dataset') or 'global_mmlu'}_oracle",
                "benchmark_content_accessed": True,
                "uses_gold_answer": True,
                "can_be_used_for_clean_eval": False,
            }
        )
        concept_index.append(
            {
                "concept_id": concept_id,
                "subject": card.get("subject"),
                "index_key": "\n".join(
                    [
                        f"Subject: {card.get('subject') or ''}",
                        f"Concept: {card.get('concept') or ''}",
                        f"Description: {card.get('description') or ''}",
                    ]
                ),
                "bank_component": card.get("bank_component") or f"{card.get('source_dataset') or 'global_mmlu'}_oracle",
            }
        )
    return registry, concept_index


def format_usage_index_text(card: dict[str, Any]) -> tuple[str, str]:
    index_key = "\n".join(
        [
            f"Subject: {card.get('subject') or ''}",
            f"Concept: {card.get('concept') or ''}",
            f"Description: {card.get('description') or ''}",
            "Trigger:",
            *[f"- {item}" for item in card.get("trigger") or []],
        ]
    )
    payload = "\n".join(
        [
            "Decision:",
            *[f"- {item}" for item in card.get("decision") or []],
            "Pitfall:",
            *[f"- {item}" for item in card.get("pitfall") or []],
        ]
    )
    return index_key, payload


def build_usage_index_rows(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for card in cards:
        index_key, payload = format_usage_index_text(card)
        rows.append(
            {
                "usage_id": card.get("usage_id"),
                "usage_job_id": card.get("usage_job_id"),
                "concept_id": card.get("concept_id"),
                "subject": card.get("subject"),
                "index_key": index_key,
                "payload": payload,
                "source_sample_ids": card.get("source_sample_ids") or [],
                "bank_component": card.get("bank_component") or f"{card.get('source_dataset') or 'global_mmlu'}_oracle",
                "benchmark_content_accessed": True,
                "uses_gold_answer": True,
                "can_be_used_for_clean_eval": False,
            }
        )
    return rows


def runtime_description(card: dict[str, Any]) -> str:
    source_dataset = str(card.get("source_dataset") or "benchmark")
    source_label = {"global_mmlu": "Global-MMLU", "mmlu_prox": "MMLU-ProX"}.get(source_dataset, source_dataset)
    lines = [
        "Concept-level CA-Mem card.",
        f"Retrieval unit: one card per {source_label} English oracle question-derived concept.",
        f"Subject: {card.get('subject') or ''}.",
        f"Concept: {card.get('concept') or ''}.",
        f"Concept ID: {card.get('concept_id') or ''}.",
    ]
    lines.extend(
        [
            f"Description: {card.get('description') or ''}",
            "Trigger:",
            *[f"- {item}" for item in card.get("trigger") or []],
            "Decision:",
            *[f"- {item}" for item in card.get("decision") or []],
            "Pitfall:",
            *[f"- {item}" for item in card.get("pitfall") or []],
            f"Provenance: {source_label} English oracle card; diagnostic-only.",
        ]
    )
    return "\n".join(lines).strip()


def write_runtime_bank(output_dir: Path, cards: list[dict[str, Any]], *, build_embedding_index: bool, args: argparse.Namespace) -> dict[str, Any]:
    bank_dir = output_dir / "banks" / f"concept_card_qwen3_{args.embedding_truncate_dim}"
    rows = []
    for card in cards:
        rows.append(
            {
                "memory_id": stable_id("mem_concept", card.get("usage_id"), card.get("concept_id")),
                "usage_id": card.get("usage_id"),
                "concept_id": card.get("concept_id"),
                "subject": card.get("subject"),
                "concept": card.get("concept"),
                "description": runtime_description(card),
                "source_sample_ids": card.get("source_sample_ids") or [],
                "bank_component": card.get("bank_component") or f"{card.get('source_dataset') or 'global_mmlu'}_oracle",
                "benchmark_content_accessed": True,
                "uses_gold_answer": True,
                "can_be_used_for_clean_eval": False,
            }
        )
    write_jsonl(bank_dir / "bank.jsonl", rows)
    meta = {
        "bank_dir": str(bank_dir),
        "card_count": len(rows),
        "bank_jsonl": str(bank_dir / "bank.jsonl"),
        "build_embedding_index": bool(build_embedding_index),
        "embedding_index": None,
    }
    if build_embedding_index and rows:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(args.embedding_model_dir), device=args.embedding_device)
        embeddings = []
        texts = [row["description"] for row in rows]
        for start in range(0, len(texts), int(args.embedding_batch_size)):
            batch = texts[start : start + int(args.embedding_batch_size)]
            kwargs = {"normalize_embeddings": True, "convert_to_numpy": True}
            try:
                vectors = model.encode(batch, prompt_name="document", truncate_dim=int(args.embedding_truncate_dim), **kwargs)
            except TypeError:
                vectors = model.encode(batch, **kwargs)
                if vectors.shape[1] > int(args.embedding_truncate_dim):
                    vectors = vectors[:, : int(args.embedding_truncate_dim)]
                    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            embeddings.append(np.asarray(vectors, dtype=np.float32))
        matrix = np.concatenate(embeddings, axis=0)
        np.save(bank_dir / "build_index.npy", matrix)
        meta["embedding_index"] = str(bank_dir / "build_index.npy")
        meta["embedding_shape"] = list(matrix.shape)
    (bank_dir / "runtime_bank_manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return meta


def build_manifest(
    args: argparse.Namespace,
    *,
    rows: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    runtime_meta: dict[str, Any] | None,
    model_usage: dict[str, int],
    errors: int,
) -> dict[str, Any]:
    subjects = sorted({str(row.get("subject") or "") for row in rows})
    by_subject = Counter(str(card.get("subject") or "") for card in cards)
    source_dataset = str(args.source_dataset)
    language = str(args.source_language or "en")
    return {
        "bank_version": args.bank_version,
        "construction_mode": construction_mode(args),
        "construction_model": "dry_run" if args.dry_run else (args.model if args.backend == "api" else str(args.local_model_dir)),
        "construction_backend": "dry_run" if args.dry_run else args.backend,
        "source_dataset": source_dataset,
        "source_dataset_label": dataset_label(args),
        "source_language": language,
        "source_input": str(args.input_jsonl),
        "source_question_count": len(rows),
        "subjects": len(subjects),
        "subject_ids": subjects,
        "oracle_node_count": len(cards),
        "concept_count": len({card.get("concept_id") for card in cards}),
        "active_card_count": len(cards),
        "usage_card_count": len(cards),
        "usage_claim_count": len(claims),
        "usage_index_count": len(cards),
        "by_subject_card_count": dict(sorted(by_subject.items())),
        "benchmark_content_accessed": True,
        "uses_gold_answer": True,
        "can_be_used_for_clean_eval": False,
        "can_be_used_for_clean_global_mmlu_eval": False,
        "exclude_global_mmlu_overlap": bool(args.exclude_global_mmlu_overlap),
        "provenance_warning": provenance_warning(args),
        "runtime_bank": runtime_meta,
        "model_usage": model_usage,
        "errors": errors,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_summary_and_report(output_dir: Path, manifest: dict[str, Any]) -> None:
    summary = {
        "ok": True,
        "output_dir": str(output_dir),
        "construction_mode": manifest["construction_mode"],
        "source_question_count": manifest["source_question_count"],
        "usage_card_count": manifest["usage_card_count"],
        "usage_claim_count": manifest["usage_claim_count"],
        "subjects": manifest["subjects"],
        "benchmark_content_accessed": True,
        "uses_gold_answer": True,
        "can_be_used_for_clean_eval": False,
        "errors": manifest["errors"],
        "runtime_bank": manifest.get("runtime_bank"),
        "model_usage": manifest.get("model_usage"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# {manifest.get('source_dataset_label') or manifest.get('source_dataset') or 'Benchmark'} English Oracle Usage Cards",
        "",
        str(manifest.get("provenance_warning") or "Diagnostic/oracle bank derived from English questions and verified answer text."),
        "",
        f"- source questions: {manifest['source_question_count']}",
        f"- usage cards: {manifest['usage_card_count']}",
        f"- subjects: {manifest['subjects']}",
        f"- errors: {manifest['errors']}",
        f"- construction model: {manifest['construction_model']}",
        f"- output_dir: `{output_dir}`",
        "",
        "This bank is not clean-evaluation safe: `uses_gold_answer=true`.",
    ]
    runtime = manifest.get("runtime_bank") or {}
    if runtime:
        lines.extend(["", "## Runtime Bank", "", f"- bank_jsonl: `{runtime.get('bank_jsonl')}`", f"- embedding_index: `{runtime.get('embedding_index')}`"])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        reset_outputs(output_dir)
    rows = select_rows(args)
    plan = {
        "input_jsonl": str(args.input_jsonl),
        "output_dir": str(output_dir),
        "source_dataset": str(args.source_dataset),
        "source_language": str(args.source_language or "en"),
        "dataset_label": dataset_label(args),
        "bank_component": bank_component(args),
        "namespace_salt": str(args.namespace_salt or ""),
        "rows": len(rows),
        "subjects": args.subjects,
        "source_splits": args.source_splits,
        "exclude_global_mmlu_overlap": bool(args.exclude_global_mmlu_overlap),
        "max_rows": args.max_rows,
        "max_rows_per_subject": args.max_rows_per_subject,
        "cards_per_question": args.cards_per_question,
        "model": "dry_run" if args.dry_run else (args.model if args.backend == "api" else str(args.local_model_dir)),
        "backend": "dry_run" if args.dry_run else args.backend,
        "max_model_calls": args.max_model_calls,
        "resume": args.resume,
        "benchmark_content_accessed": True,
        "uses_gold_answer": True,
        "can_be_used_for_clean_eval": False,
    }
    write_jsonl(output_dir / "plan.jsonl", [plan])
    existing_ok = existing_eval_keys(output_dir) if args.resume else set()
    pending = [row for row in rows if row["_eval_key"] not in existing_ok]
    estimated_calls = 0 if args.dry_run else len(pending)
    if estimated_calls > int(args.max_model_calls):
        raise RuntimeError(f"refusing run: estimated model calls {estimated_calls} > max {args.max_model_calls}")

    client: Any | None = None
    if not args.dry_run and pending:
        if args.backend == "api":
            model_config = ModelConfig(
                name=args.model,
                reasoning_effort=args.reasoning_effort,
                max_completion_tokens=args.max_completion_tokens,
                concurrency=args.concurrency,
                max_retries=args.max_retries,
                # ModelConfig is smoke-test scoped and caps this field at 800.
                # This script enforces its own full-run budget with max_model_calls above.
                max_total_calls=min(max(estimated_calls, 1), 800),
            )
            client = SmokeModelClient.from_config(
                model_config,
                cache_dir=output_dir / "cache" / "model",
                raw_dir=output_dir / "logs" / "model_raw",
                timeout_s=args.model_timeout_s,
            )
        else:
            from scripts.run_multilingual_rag_baselines import LocalVLLMModel

            client = LocalVLLMModel(
                model_dir=args.local_model_dir,
                max_model_len=args.local_max_model_len,
                gpu_memory_utilization=args.local_gpu_memory_utilization,
                batch_max_tokens=args.local_batch_max_tokens,
                local_batch_size=args.local_batch_size,
                local_batch_timeout_ms=args.local_batch_timeout_ms,
                cache_dir=output_dir / "cache" / "local_vllm",
            )

    append_jsonl(output_dir / "build_events.jsonl", [{"event": "start", **plan, "pending": len(pending)}])
    if not args.resume:
        append_jsonl(output_dir / "source_items.jsonl", [build_source_item(row) for row in rows])
    else:
        source_path = output_dir / "source_items.jsonl"
        if not source_path.exists():
            append_jsonl(source_path, [build_source_item(row) for row in rows])

    try:
        completed = len(existing_ok)
        errors = 0
        for batch_start in range(0, len(pending), int(args.batch_size)):
            batch = pending[batch_start : batch_start + int(args.batch_size)]
            tasks = [
                build_one(
                    row,
                    client=client,
                    backend=str(args.backend),
                    cards_per_question=int(args.cards_per_question),
                    dry_run=bool(args.dry_run),
                    max_completion_tokens=int(args.max_completion_tokens),
                    max_retries=int(args.max_retries),
                )
                for row in batch
            ]
            outputs = await asyncio.gather(*tasks)
            new_cards: list[dict[str, Any]] = []
            raw_cards: list[dict[str, Any]] = []
            logs: list[dict[str, Any]] = []
            error_rows: list[dict[str, Any]] = []
            for row, (cards, result_log, error) in zip(batch, outputs):
                ok = not error and bool(cards)
                completed += int(ok)
                errors += int(not ok)
                for card in cards:
                    raw_cards.append(card)
                    new_cards.append(card)
                logs.append(
                    {
                        "eval_key": row["_eval_key"],
                        "sample_id": row.get("sample_id"),
                        "subject": row.get("subject"),
                        "ok": ok,
                        "card_count": len(cards),
                        "error": error,
                        **result_log,
                    }
                )
                if not ok:
                    error_rows.append(
                        {
                            "stage": "oracle_card_extraction",
                            "eval_key": row["_eval_key"],
                            "sample_id": row.get("sample_id"),
                            "subject": row.get("subject"),
                            "error": error or "unknown_error",
                        }
                    )
            append_jsonl(output_dir / "usage_cards.raw.jsonl", raw_cards)
            append_jsonl(output_dir / "usage_cards.jsonl", new_cards)
            append_jsonl(output_dir / "usage_card_claims.jsonl", build_claim_rows(new_cards))
            append_jsonl(output_dir / "card_extraction_logs.jsonl", logs)
            append_jsonl(output_dir / "errors.jsonl", error_rows)
            if args.progress_every and completed % int(args.progress_every) < len(batch):
                print(json.dumps({"event": "progress", "completed_ok": completed, "errors": errors, "pending_total": len(pending)}, ensure_ascii=False), flush=True)
    finally:
        if client is not None:
            await client.aclose()

    cards = read_jsonl(output_dir / "usage_cards.jsonl") if (output_dir / "usage_cards.jsonl").exists() else []
    claims = read_jsonl(output_dir / "usage_card_claims.jsonl") if (output_dir / "usage_card_claims.jsonl").exists() else []
    registry, concept_index = build_concept_rows(cards)
    usage_index = build_usage_index_rows(cards)
    write_jsonl(output_dir / "concept_registry.jsonl", registry)
    write_jsonl(output_dir / "concept_index.jsonl", concept_index)
    write_jsonl(output_dir / "usage_index.jsonl", usage_index)
    write_jsonl(output_dir / "usage_consolidation.jsonl", [])
    write_jsonl(output_dir / "rejected_items.jsonl", [])

    runtime_meta = None
    if args.write_runtime_bank:
        runtime_meta = write_runtime_bank(output_dir, cards, build_embedding_index=bool(args.build_embedding_index), args=args)
    logs = read_jsonl(output_dir / "card_extraction_logs.jsonl") if (output_dir / "card_extraction_logs.jsonl").exists() else []
    error_count = sum(1 for row in logs if not row.get("ok"))
    if client is None:
        model_usage = {"network_model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    elif hasattr(client, "usage_summary"):
        model_usage = client.usage_summary()
    else:
        model_usage = {"local_vllm_calls": int(getattr(client, "calls", 0))}
    manifest = build_manifest(
        args,
        rows=rows,
        cards=cards,
        claims=claims,
        runtime_meta=runtime_meta,
        model_usage=model_usage,
        errors=error_count,
    )
    (output_dir / "bank_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_and_report(output_dir, manifest)
    append_jsonl(output_dir / "build_events.jsonl", [{"event": "finish", "summary": manifest}])
    return manifest


def main() -> int:
    args = parse_args()
    manifest = asyncio.run(run(args))
    print(json.dumps({"ok": True, "output_dir": str(args.output_dir), "usage_card_count": manifest["usage_card_count"], "errors": manifest["errors"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
