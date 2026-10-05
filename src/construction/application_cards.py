"""Synthesize source-grounded applications before extracting five-field cards.

Application questions are intermediate construction records. The exported card
body and runtime description use the existing interfaces.
"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import time
import unicodedata
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.construction.model_client import ApplicationModelClient
from src.construction.io import write_json, write_jsonl


VERSION = "wikipag-application-cards-v3"
PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts" / "wikipag_application"
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CardField = Literal["concept_name", "definition", "trigger", "decision", "pitfall"]


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Passage(StrictRecord):
    source_id: Text
    passage_id: Text
    title: str = ""
    text: Annotated[Text, Field(max_length=12000)]


class PassageGroup(StrictRecord):
    group_id: Text
    subject: Text
    corpus: Literal["English Wikipag"]
    language: Literal["en"]
    passages: Annotated[list[Passage], Field(min_length=1, max_length=5)]

    @model_validator(mode="after")
    def validate_passages(self) -> "PassageGroup":
        keys = [(p.source_id, p.passage_id) for p in self.passages]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate passage identity within group")
        if sum(len(p.text) for p in self.passages) > 30000:
            raise ValueError("passage group exceeds 30000 characters; split it explicitly")
        return self


class Citation(StrictRecord):
    source_id: Text
    passage_id: Text
    quote: Text


class Application(StrictRecord):
    question: Text
    assumptions: list[Text]
    evidence: Annotated[list[Citation], Field(min_length=1)]


class Solution(StrictRecord):
    answer: Text
    explanation: Text
    evidence: Annotated[list[Citation], Field(min_length=1)]


class SelfCheck(StrictRecord):
    requires_application: bool
    source_sufficient: bool
    assumptions_explicit: bool
    answer_correct: bool
    english_only: bool
    issues: list[Text]
    evidence: Annotated[list[Citation], Field(min_length=1)]

    @property
    def accepted(self) -> bool:
        return all((self.requires_application, self.source_sufficient,
                    self.assumptions_explicit, self.answer_correct, self.english_only)) and not self.issues


class Card(StrictRecord):
    # Keep the current cards.jsonl contract, including an optional empty pitfall list.
    concept_name: Text
    definition: Text
    trigger: Annotated[list[Text], Field(min_length=1)]
    decision: Annotated[list[Text], Field(min_length=1)]
    pitfall: list[Text]


class Grounding(StrictRecord):
    field: CardField
    item_index: Annotated[int, Field(ge=0)] | None
    evidence: Annotated[list[Citation], Field(min_length=1)]
    support_type: Literal["direct", "derived"]
    rationale: Text


class GroundedCard(StrictRecord):
    card: Card
    grounding: list[Grounding]

    @model_validator(mode="after")
    def cover_every_field(self) -> "GroundedCard":
        expected = {("concept_name", None), ("definition", None)}
        for name in ("trigger", "decision", "pitfall"):
            expected.update((name, i) for i in range(len(getattr(self.card, name))))
        actual = [(g.field, g.item_index) for g in self.grounding]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            label = lambda key: f"{key[0]}[{key[1] if key[1] is not None else 'null'}]"
            missing = sorted(label(key) for key in expected - set(actual))
            extra = sorted(label(key) for key in set(actual) - expected)
            repeated = sorted(label(key) for key in set(actual) if actual.count(key) > 1)
            raise ValueError("grounding must cover each scalar and list item exactly once; "
                             f"missing={missing}, extra={extra}, repeated={repeated}")
        return self


class Extraction(StrictRecord):
    cards: Annotated[list[GroundedCard], Field(max_length=3)]


class CardJudgment(StrictRecord):
    card_index: Annotated[int, Field(ge=0)]
    source_supported: bool
    conditions_preserved: bool
    reusable: bool
    useful_application: bool
    solution_steps_preserved: bool
    english_only: bool
    missing_steps: list[Text]
    issues: list[Text]
    evidence: Annotated[list[Citation], Field(min_length=1)]

    @property
    def accepted(self) -> bool:
        return all((self.source_supported, self.conditions_preserved, self.reusable,
                    self.useful_application, self.solution_steps_preserved, self.english_only)) \
            and not self.issues and not self.missing_steps


class CardReview(StrictRecord):
    cards: Annotated[list[CardJudgment], Field(max_length=3)]

    def cover_candidates(self, count: int) -> None:
        indexes = [c.card_index for c in self.cards]
        if sorted(indexes) != list(range(count)):
            raise ValueError("card review must cover every candidate exactly once")


class CardReplacement(StrictRecord):
    card_index: Annotated[int, Field(ge=0)]
    replacement: GroundedCard | None
    reason: Text


class CardRepair(StrictRecord):
    repairs: Annotated[list[CardReplacement], Field(min_length=1, max_length=3)]


GENERATION_STAGES = ("application", "solution", "self_check", "extraction")
SCHEMAS = {"application": Application, "solution": Solution,
           "self_check": SelfCheck, "extraction": Extraction, "card_review": CardReview,
           "card_repair": CardRepair, "repair_review": CardReview}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def load_groups(path: Path) -> list[PassageGroup]:
    groups = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    groups.append(PassageGroup.model_validate(json.loads(line)))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"invalid passage group at line {line_no}: {exc}") from exc
    validate_groups(groups)
    return groups


def validate_groups(groups: list[PassageGroup]) -> None:
    if not groups:
        raise ValueError("at least one English Wikipag passage group is required")
    ids: set[str] = set()
    passages: dict[tuple[str, str], str] = {}
    for group in groups:
        if group.group_id in ids:
            raise ValueError(f"duplicate group_id: {group.group_id}")
        ids.add(group.group_id)
        for passage in group.passages:
            key = (passage.source_id, passage.passage_id)
            if key in passages and passages[key] != passage.text:
                raise ValueError(f"conflicting texts for passage: {key}")
            passages[key] = passage.text


def validate_payload(stage: str, payload: dict[str, Any], group: PassageGroup,
                     context: dict[str, Any] | None = None) -> StrictRecord:
    value = SCHEMAS[stage].model_validate(payload)
    sources = {(p.source_id, p.passage_id): p.text for p in group.passages}
    if isinstance(value, Extraction):
        citations = [c for card in value.cards for g in card.grounding for c in g.evidence]
    elif isinstance(value, CardReview):
        if context is None:
            raise ValueError("card review requires candidate context")
        value.cover_candidates(len(Extraction.model_validate(context["extraction"]).cards))
        citations = [c for judgment in value.cards for c in judgment.evidence]
    elif isinstance(value, CardRepair):
        if context is None:
            raise ValueError("card repair requires review context")
        review = CardReview.model_validate(context["card_review"])
        review.cover_candidates(len(Extraction.model_validate(context["extraction"]).cards))
        expected = sorted(c.card_index for c in review.cards if not c.accepted)
        if sorted(r.card_index for r in value.repairs) != expected:
            raise ValueError("repair must cover exactly the rejected candidates")
        citations = [c for repair in value.repairs if repair.replacement is not None
                     for g in repair.replacement.grounding for c in g.evidence]
    else:
        citations = list(value.evidence)
    for citation in citations:
        text = sources.get((citation.source_id, citation.passage_id))
        if text is None or citation.quote not in text:
            raise ValueError("citation must quote an exact span from an identified input passage; "
                             f"source_id={citation.source_id!r}, passage_id={citation.passage_id!r}, "
                             f"quote={citation.quote[:240]!r}; copy original text without shortening or replacing punctuation")
    return value


def render_card(card: Card) -> str:
    return "\n".join([
        f"Description: {card.definition}", "Trigger:",
        *[f"- {text}" for text in card.trigger], "Decision:",
        *[f"- {text}" for text in card.decision], "Pitfall:",
        *[f"- {text}" for text in card.pitfall],
    ])


def content_key(subject: str, card: Card) -> str:
    # Preserve case, symbols, conditions and item order. Name alone cannot merge rules.
    def normalize(value: Any) -> Any:
        if isinstance(value, str):
            return " ".join(unicodedata.normalize("NFC", value).split())
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return {k: normalize(v) for k, v in value.items()}
    return digest({"subject": subject, "card": normalize(card.model_dump())})


class StageError(RuntimeError):
    pass


class ApplicationCardBuilder:
    def __init__(self, client: ApplicationModelClient, output_dir: Path, *, provider_id: str = "configured",
                 preparer: Any = None, max_card_repairs: int = 1):
        from src.construction.source_preparation import SourcePreparer
        if max_card_repairs not in (0, 1):
            raise ValueError("max_card_repairs must be 0 or 1")
        self.client = client
        self.output_dir = output_dir
        self.preparer = preparer or SourcePreparer()
        self.max_card_repairs = max_card_repairs
        self.prompts = {stage: (PROMPT_DIR / f"{'card_review' if stage == 'repair_review' else stage}.txt").read_text(encoding="utf-8")
                        for stage in SCHEMAS}
        self.protocol = {
            "version": VERSION,
            "model": client.model_config.model_dump(mode="json", exclude={"concurrency"}),
            "provider_id": provider_id,
            "prompt_sha256": {s: digest(p) for s, p in self.prompts.items()},
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "support_code_sha256": {str(path.relative_to(Path(__file__).resolve().parents[2])):
                                    hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in [Path(__file__).with_name(name) for name in
                                                 ("config.py", "model_client.py", "io.py")]
                                    + [Path(__file__).resolve().parents[1] / "clients" / "cache.py"]},
            "source_preparation": self.preparer.protocol,
            "max_card_repairs": max_card_repairs,
        }

    async def stage(self, stage: str, group: PassageGroup, context: dict[str, Any]) -> StrictRecord:
        schema = SCHEMAS[stage]
        messages = [
            {"role": "system", "content": self.prompts[stage]},
            {"role": "user", "content": json.dumps({
                "passage_group": group.model_dump(), "construction_record": context,
                "required_output_schema": schema.model_json_schema(),
            }, ensure_ascii=False)},
        ]
        request_hash = digest({"protocol": self.protocol, "stage": stage, "messages": messages})
        path = self.output_dir / "stages" / digest(group.group_id) / f"{stage}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("request_sha256") != request_hash:
                raise StageError(f"checkpoint mismatch: {path}; use a new run directory")
            if saved.get("payload_sha256") != digest(saved["payload"]):
                raise StageError(f"checkpoint payload checksum mismatch: {path}")
            return validate_payload(stage, saved["payload"], group, context)
        value, result, error = await self.client.complete_json(
            messages, namespace=f"wikipag.{stage}.{digest(group.group_id)}",
            validate=lambda payload: validate_payload(stage, payload, group, context),
        )
        if value is None:
            raise StageError(f"{stage}: {error or 'model request failed'}")
        # Validate even injected/test clients at the stage boundary.
        value = validate_payload(stage, value.model_dump(mode="json"), group, context)
        payload = value.model_dump(mode="json")
        write_json(path, {"request_sha256": request_hash, "payload": payload, "payload_sha256": digest(payload),
                          "response_cache_key": getattr(result, "cache_key", None),
                          "model_trace": {"latency_s": getattr(result, "latency_s", None),
                                          "usage": getattr(result, "usage", {}),
                                          "cached": getattr(result, "cached", False),
                                          "served_model": getattr(result, "raw", {}).get("model")}})
        return value

    async def process_group(self, group: PassageGroup) -> dict[str, Any]:
        context: dict[str, Any] = {}
        record: dict[str, Any] = {"group_id": group.group_id, "subject": group.subject,
                                  "stage_directory": f"stages/{digest(group.group_id)}",
                                  "preparation_file": f"source_preparation/{digest(group.group_id)}.json"}
        try:
            group, preparation = self.preparer.prepare(group, self.output_dir)
            record["source_preparation"] = preparation
            if group is None:
                return {**record, "status": "rejected", "rejection_stage": "source_preparation", "records": context}
            record["passage_group"] = group.model_dump(mode="json")
            for stage in GENERATION_STAGES:
                value = await self.stage(stage, group, context)
                context[stage] = value.model_dump(mode="json")
                if stage == "self_check" and not value.accepted:
                    return {**record, "status": "rejected", "rejection_stage": stage, "records": context}
            if not context["extraction"]["cards"]:
                return {**record, "status": "rejected", "rejection_stage": "empty_extraction", "records": context}
            review_context = {key: context[key] for key in ("application", "solution", "extraction")}
            review = await self.stage("card_review", group, review_context)
            context["card_review"] = review.model_dump(mode="json")
            record.update(export_extraction="extraction", export_review="card_review",
                          candidate_origins=list(range(len(context["extraction"]["cards"]))))
            if self.max_card_repairs and any(not c.accepted for c in review.cards):
                repair = await self.stage("card_repair", group, {**review_context, "card_review": context["card_review"]})
                context["card_repair"] = repair.model_dump(mode="json")
                replacements = {r.card_index: r.replacement for r in repair.repairs}
                candidates = Extraction.model_validate(context["extraction"]).cards
                retained = [(i, replacements.get(i, c)) for i, c in enumerate(candidates)]
                retained = [(i, c) for i, c in retained if c is not None]
                if not retained:
                    return {**record, "status": "rejected", "rejection_stage": "card_repair", "records": context}
                revised = Extraction(cards=[c for _, c in retained]).model_dump(mode="json")
                context["repaired_extraction"] = revised
                # The second reviewer sees the revised cards without the first review's verdict.
                review = await self.stage("repair_review", group, {**review_context, "extraction": revised})
                context["repair_review"] = review.model_dump(mode="json")
                record.update(export_extraction="repaired_extraction", export_review="repair_review",
                              candidate_origins=[i for i, _ in retained])
            accepted = any(c.accepted for c in review.cards)
            return {**record, "status": "accepted" if accepted else "rejected",
                    "rejection_stage": None if accepted else record["export_review"], "records": context}
        except (StageError, ValueError, KeyError, OSError) as exc:
            return {**record, "status": "error", "error": str(exc), "records": context}

    async def run(self, groups: list[PassageGroup]) -> dict[str, Any]:
        validate_groups(groups)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / ".construction.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("another builder is writing this run directory") from exc
            return await self._run_locked(groups)

    async def _run_locked(self, groups: list[PassageGroup]) -> dict[str, Any]:
        started = time.monotonic()
        inputs = [g.model_dump(mode="json") for g in groups]
        manifest = {**self.protocol, "input_sha256": digest(inputs), "groups": len(groups),
                    "source_corpus": "English Wikipag", "language": "en"}
        manifest_path = self.output_dir / "construction_manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("input/model/prompt configuration changed; use a new run directory")
        else:
            # The transport can create its empty cache/raw directories before run().
            allowed = {".construction.lock", "cache", "model_raw"}
            if any(p.name not in allowed for p in self.output_dir.iterdir()):
                raise ValueError("output directory contains another run; choose an empty directory")
            write_json(manifest_path, manifest)
        write_jsonl(self.output_dir / "passage_groups.jsonl", inputs)
        write_json(self.output_dir / "summary.json", {"complete": False, "groups": len(groups)})
        queue: asyncio.Queue[tuple[int, PassageGroup]] = asyncio.Queue()
        for index, group in enumerate(groups):
            queue.put_nowait((index, group))
        results: list[dict[str, Any] | None] = [None] * len(groups)

        async def worker() -> None:
            while not queue.empty():
                index, group = queue.get_nowait()
                results[index] = await self.process_group(group)
                queue.task_done()
                write_json(self.output_dir / "progress.json", {
                    "groups": len(groups), "finished": sum(r is not None for r in results),
                    "accepted": sum(r is not None and r["status"] == "accepted" for r in results),
                    "rejected": sum(r is not None and r["status"] == "rejected" for r in results),
                    "errors": sum(r is not None and r["status"] == "error" for r in results),
                    "wall_time_s": time.monotonic() - started,
                })

        await asyncio.gather(*(worker() for _ in range(min(len(groups), self.client.model_config.concurrency))))
        completed = [r for r in results if r is not None]
        write_jsonl(self.output_dir / "synthesis_records.jsonl", completed)
        write_jsonl(self.output_dir / "rejected_records.jsonl", [r for r in completed if r["status"] == "rejected"])
        write_jsonl(self.output_dir / "errors.jsonl", [r for r in completed if r["status"] == "error"])
        write_jsonl(self.output_dir / "prepared_passage_groups.jsonl", [r["passage_group"] for r in completed if "passage_group" in r])
        card_rejections = [{"group_id": r["group_id"], "stage": stage, "judgment": c,
                            "stage_directory": r["stage_directory"]}
                           for r in completed for stage in ("card_review", "repair_review")
                           for c in r["records"].get(stage, {}).get("cards", [])
                           if not CardJudgment.model_validate(c).accepted]
        write_jsonl(self.output_dir / "card_rejections.jsonl", card_rejections)
        cards, metadata, runtime = self.export(completed)
        write_jsonl(self.output_dir / "cards.jsonl", cards)
        write_jsonl(self.output_dir / "bank_metadata.jsonl", metadata)
        write_jsonl(self.output_dir / "bank.jsonl", runtime)
        summary = {
            "complete": not any(r["status"] == "error" for r in completed),
            "groups": len(groups), "accepted_groups": sum(r["status"] == "accepted" for r in completed),
            "rejected_groups": sum(r["status"] == "rejected" for r in completed),
            "error_groups": sum(r["status"] == "error" for r in completed), "cards": len(cards),
            "groups_with_repair": sum("card_repair" in r["records"] for r in completed),
            "rejected_card_reviews": len(card_rejections),
            "filtered_seed_passages": sum(r.get("source_preparation", {}).get("filtered_seeds", 0) for r in completed),
            "supplementary_passages": sum(r.get("source_preparation", {}).get("supplementary_passages", 0) for r in completed),
            "model": self.client.model_config.name, "source_corpus": "English Wikipag",
            "card_fields": list(Card.model_fields), "vectors_built": False,
            "wall_time_s": time.monotonic() - started,
            "execution_model_usage": self.client.usage_summary() if hasattr(self.client, "usage_summary") else None,
            "validation": "Source heuristics, JSON schemas, exact quotations, field/review coverage, same-model application and card review; not independent correctness verification",
            "output_sha256": {name: hashlib.sha256((self.output_dir / name).read_bytes()).hexdigest()
                              for name in ["cards.jsonl", "bank_metadata.jsonl", "bank.jsonl", "synthesis_records.jsonl"]},
        }
        write_json(self.output_dir / "summary.json", summary)
        return summary

    def export(self, results: list[dict[str, Any]]) -> tuple[list[dict], list[dict], list[dict]]:
        cards: list[dict] = []
        metadata: list[dict] = []
        runtime: list[dict] = []
        seen: dict[str, int] = {}
        for record in results:
            if record["status"] != "accepted":
                continue
            extraction = Extraction.model_validate(record["records"][record["export_extraction"]])
            review = CardReview.model_validate(record["records"][record["export_review"]])
            review.cover_candidates(len(extraction.cards))
            judgments = {c.card_index: c for c in review.cards}
            for candidate_index, candidate in enumerate(extraction.cards):
                if not judgments[candidate_index].accepted:
                    continue
                card = candidate.card
                key = content_key(record["subject"], card)
                origin = {"group_id": record["group_id"], "stage_directory": record["stage_directory"],
                          "candidate_index": candidate_index,
                          "original_candidate_index": record["candidate_origins"][candidate_index],
                          "extraction_stage": record["export_extraction"], "review_stage": record["export_review"],
                          "review": judgments[candidate_index].model_dump(mode="json"),
                          "preparation_file": record["preparation_file"],
                          "grounding": [g.model_dump(mode="json") for g in candidate.grounding]}
                if key in seen:
                    metadata[seen[key]]["synthesis_records"].append(origin)
                    continue
                index = len(cards)
                seen[key] = index
                memory_id = f"mem_wikipag_{key}"
                cards.append(card.model_dump(mode="json"))
                metadata.append({"index": index, "memory_id": memory_id, "subject": record["subject"],
                                 "source_corpus": "English Wikipag", "bank_version": VERSION,
                                 "synthesis_records": [origin]})
                runtime.append({"memory_id": memory_id, "subject": record["subject"],
                                "concept": card.concept_name, "description": render_card(card)})
        return cards, metadata, runtime
