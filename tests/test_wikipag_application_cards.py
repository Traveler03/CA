from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.construction.application_cards import (
    ApplicationCardBuilder, Card, PassageGroup, digest, load_groups, validate_payload,
)
from src.construction.config import ModelConfig
from src.construction.io import read_jsonl


# Deliberately small in-memory fixtures; these are not corpus assets or measured results.
SOURCE = "Speed is distance divided by elapsed time. Average speed uses total distance and total elapsed time."


def group(group_id="group-1"):
    return PassageGroup.model_validate({
        "group_id": group_id, "subject": "physics", "corpus": "English Wikipag", "language": "en",
        "passages": [{"source_id": "fixture-source", "passage_id": "fixture-passage", "text": SOURCE}],
    })


def responses():
    citation = {"source_id": "fixture-source", "passage_id": "fixture-passage",
                "quote": "Speed is distance divided by elapsed time."}
    card = {"concept_name": "Average speed", "definition": "Total distance divided by total elapsed time.",
            "trigger": ["Distance and elapsed time are known."],
            "decision": ["Divide total distance by total elapsed time."], "pitfall": []}
    grounding = [{"field": name, "item_index": index, "evidence": [deepcopy(citation)],
                  "support_type": "direct", "rationale": "The passage states the distance/time relationship."}
                 for name, index in [("concept_name", None), ("definition", None), ("trigger", 0), ("decision", 0)]]
    return {
        "application": {"question": "What is the average speed of a vehicle travelling 120 km in 2 hours?",
                        "assumptions": ["The elapsed time includes all stops."], "evidence": [citation]},
        "solution": {"answer": "60 km/h", "explanation": "Divide 120 km by 2 hours to obtain 60 km/h.",
                     "evidence": [citation]},
        "self_check": {"requires_application": True, "source_sufficient": True, "assumptions_explicit": True,
                       "answer_correct": True, "english_only": True, "issues": [], "evidence": [citation]},
        "extraction": {"cards": [{"card": card, "grounding": grounding}]},
        "card_review": {"cards": [{"card_index": 0, "source_supported": True, "conditions_preserved": True,
                                  "reusable": True, "useful_application": True, "solution_steps_preserved": True,
                                  "english_only": True, "missing_steps": [], "issues": [], "evidence": [citation]}]},
    }


class FakeClient:
    def __init__(self, payloads=None, fail_stage=None, model="fixture-model"):
        self.model_config = ModelConfig(name=model, max_retries=0, concurrency=2)
        self.payloads = responses() if payloads is None else payloads
        self.fail_stage = fail_stage
        self.calls = []

    async def complete_json(self, messages, *, namespace, validate):
        stage = namespace.split(".")[1]
        request = json.loads(messages[-1]["content"])
        self.calls.append((stage, request))
        if stage == self.fail_stage:
            return None, None, "transient fixture failure"
        return validate(deepcopy(self.payloads[stage])), SimpleNamespace(cache_key=f"fixture-{stage}"), None


def build(client, output, groups=None):
    return asyncio.run(ApplicationCardBuilder(client, output).run(groups or [group()]))


def test_exports_existing_body_and_runtime_format_with_separate_trace(tmp_path):
    client = FakeClient()
    result = build(client, tmp_path)
    assert result["complete"] and result["cards"] == 1
    assert [stage for stage, _ in client.calls] == ["application", "solution", "self_check", "extraction", "card_review"]
    body = read_jsonl(tmp_path / "cards.jsonl")[0]
    assert set(body) == {"concept_name", "definition", "trigger", "decision", "pitfall"}
    assert "60 km/h" not in json.dumps(body)
    runtime = read_jsonl(tmp_path / "bank.jsonl")[0]
    assert set(runtime) == {"memory_id", "subject", "concept", "description"}
    assert runtime["description"].startswith("Description: Total distance")
    assert "\nTrigger:\n- " in runtime["description"]
    assert "\nDecision:\n- " in runtime["description"]
    metadata = read_jsonl(tmp_path / "bank_metadata.jsonl")[0]
    assert metadata["memory_id"] == runtime["memory_id"]
    origin = metadata["synthesis_records"][0]
    assert origin["group_id"] == "group-1"
    assert origin["grounding"][0]["evidence"][0]["passage_id"] == "fixture-passage"
    assert (tmp_path / origin["stage_directory"] / "solution.json").exists()


@pytest.mark.parametrize("failure", ["answer_correct", "source_sufficient", "english_only", "requires_application", "issues"])
def test_failed_check_never_reaches_extraction(tmp_path, failure):
    payloads = responses()
    payloads["self_check"][failure] = ["A required assumption is missing."] if failure == "issues" else False
    client = FakeClient(payloads)
    summary = build(client, tmp_path)
    assert summary["complete"] and summary["rejected_groups"] == 1 and summary["cards"] == 0
    assert len(client.calls) == 3
    assert read_jsonl(tmp_path / "cards.jsonl") == []
    assert len(read_jsonl(tmp_path / "rejected_records.jsonl")) == 1


def test_resume_uses_stage_checkpoints_and_retries_only_unfinished_work(tmp_path):
    first = FakeClient(fail_stage="solution")
    assert build(first, tmp_path)["complete"] is False
    second = FakeClient()
    assert build(second, tmp_path)["complete"] is True
    assert [stage for stage, _ in second.calls] == ["solution", "self_check", "extraction", "card_review"]
    third = FakeClient()
    assert build(third, tmp_path)["cards"] == 1
    assert third.calls == []
    assert read_jsonl(tmp_path / "errors.jsonl") == []


def test_changed_input_or_model_cannot_reuse_an_old_run(tmp_path):
    build(FakeClient(), tmp_path)
    original = (tmp_path / "cards.jsonl").read_bytes()
    with pytest.raises(ValueError, match="configuration changed"):
        build(FakeClient(model="different-model"), tmp_path)
    changed = group().model_copy(update={"subject": "different-subject"})
    with pytest.raises(ValueError, match="configuration changed"):
        build(FakeClient(), tmp_path, [changed])
    assert (tmp_path / "cards.jsonl").read_bytes() == original


def test_tampered_checkpoint_is_not_consumed(tmp_path):
    build(FakeClient(), tmp_path)
    checkpoint = tmp_path / "stages" / digest("group-1") / "solution.json"
    saved = json.loads(checkpoint.read_text())
    saved["payload"]["answer"] = "tampered"
    checkpoint.write_text(json.dumps(saved))
    client = FakeClient()
    summary = build(client, tmp_path)
    assert not summary["complete"] and summary["error_groups"] == 1
    assert client.calls == []
    assert "checksum mismatch" in read_jsonl(tmp_path / "errors.jsonl")[0]["error"]


@pytest.mark.parametrize("failure", ["foreign_source", "invented_quote", "missing_field", "extra_answer"])
def test_unsupported_or_reshaped_card_is_rejected(tmp_path, failure):
    payloads = responses()
    candidate = payloads["extraction"]["cards"][0]
    if failure == "foreign_source":
        candidate["grounding"][0]["evidence"][0]["source_id"] = "other-source"
    elif failure == "invented_quote":
        candidate["grounding"][0]["evidence"][0]["quote"] = "The card is certainly correct."
    elif failure == "missing_field":
        candidate["grounding"].pop()
    else:
        candidate["card"]["answer"] = "60 km/h"
    summary = build(FakeClient(payloads), tmp_path)
    assert not summary["complete"] and summary["cards"] == 0


def test_identical_cards_merge_traces_but_distinct_conditions_remain(tmp_path):
    builder = ApplicationCardBuilder(FakeClient(), tmp_path)
    candidate = responses()["extraction"]
    def record(group_id, extraction):
        return {"group_id": group_id, "subject": "physics", "stage_directory": f"stages/{group_id}",
                "preparation_file": "source_preparation/fixture.json", "export_extraction": "extraction",
                "export_review": "card_review", "candidate_origins": [0],
                "status": "accepted", "records": {"extraction": extraction, "card_review": responses()["card_review"]}}
    other = deepcopy(candidate)
    other["cards"][0]["card"]["trigger"] = ["The speed is constant over the entire journey."]
    cards, metadata, runtime = builder.export([record("a", candidate), record("b", candidate), record("c", other)])
    assert len(cards) == len(metadata) == len(runtime) == 2
    assert [r["group_id"] for r in metadata[0]["synthesis_records"]] == ["a", "b"]
    assert runtime[0]["memory_id"] != runtime[1]["memory_id"]


def test_input_validation_preserves_unicode_line_separator_and_rejects_source_mismatch(tmp_path):
    row = group().model_dump()
    row["passages"][0]["text"] += "\u2028A continuation of the same passage."
    path = tmp_path / "groups.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    assert len(load_groups(path)) == 1
    row["language"] = "sw"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        load_groups(path)
    row["language"] = "en"
    conflict = deepcopy(row)
    conflict["group_id"] = "second-group"
    conflict["passages"][0]["text"] = "A different passage under the same identifier."
    path.write_text(json.dumps(row) + "\n" + json.dumps(conflict) + "\n")
    with pytest.raises(ValueError, match="conflicting texts"):
        load_groups(path)


def test_body_accepts_empty_pitfall_but_no_additional_fields():
    payload = responses()["extraction"]["cards"][0]["card"]
    assert Card.model_validate(payload).pitfall == []
    with pytest.raises(ValidationError):
        Card.model_validate({**payload, "source_id": "fixture-source"})


def test_grounding_failure_identifies_the_missing_scalar_for_correction():
    payload = responses()["extraction"]
    candidate = payload["cards"][0]
    candidate["grounding"] = [g for g in candidate["grounding"] if g["field"] != "definition"]
    with pytest.raises(ValueError, match=r"missing=\['definition\[null\]'\]"):
        validate_payload("extraction", payload, group())


def test_unknown_passage_rejected_before_downstream_stage():
    payload = responses()["application"]
    payload["evidence"][0]["passage_id"] = "missing-passage"
    with pytest.raises(ValueError, match="exact span"):
        validate_payload("application", payload, group())


def test_validate_only_cli_never_resolves_provider_or_writes_outputs(tmp_path, monkeypatch):
    from scripts import build_wikipag_application_cards as cli

    def forbidden(*args, **kwargs):
        raise AssertionError("input validation must not access credentials")

    monkeypatch.setattr(cli, "resolve_provider", forbidden)
    path = tmp_path / "inputs.jsonl"
    path.write_text(json.dumps(group().model_dump()) + "\n")
    original = path.read_bytes()
    output = tmp_path / "unused"
    assert cli.main(["--passage-groups", str(path), "--output-dir", str(output), "--validate-only"]) == 0
    assert path.read_bytes() == original
    assert not output.exists()


def test_cli_refuses_existing_bank_before_resolving_provider(tmp_path, monkeypatch):
    from scripts import build_wikipag_application_cards as cli

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("existing bank must be rejected before constructing a client")
    monkeypatch.setattr(cli, "resolve_provider", forbidden)
    output = tmp_path / "runs" / "current_bank"
    output.mkdir(parents=True)
    sentinel = output / "cards.jsonl"
    sentinel.write_text("original bank\n")
    source = tmp_path / "groups.jsonl"
    source.write_text(json.dumps(group().model_dump()) + "\n")
    with pytest.raises(ValueError, match="already contains data"):
        cli.main(["--passage-groups", str(source), "--output-dir", str(output)])
    assert sentinel.read_text() == "original bank\n"
    assert list(output.iterdir()) == [sentinel]


def test_cli_rejects_output_outside_runs(tmp_path, monkeypatch):
    from scripts import build_wikipag_application_cards as cli

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    source = tmp_path / "groups.jsonl"
    source.write_text(json.dumps(group().model_dump()) + "\n")
    with pytest.raises(ValueError, match="dedicated directory"):
        cli.main(["--passage-groups", str(source), "--output-dir", str(tmp_path / "corpus")])
    assert not (tmp_path / "corpus").exists()
