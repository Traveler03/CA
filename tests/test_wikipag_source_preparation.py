from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest

from src.construction.application_cards import ApplicationCardBuilder, Passage, PassageGroup, digest
from src.construction.source_preparation import SourcePreparer, quality_issues
from src.construction.io import read_jsonl
from test_wikipag_application_cards import FakeClient, SOURCE, group, responses


def test_missing_formula_detected_without_rejecting_intact_formula():
    damaged = Passage(source_id="s", passage_id="p", text=SOURCE + " It has a time requirement of , using big O notation.")
    assert any("missing_formula" in issue for issue in quality_issues(damaged))
    good = damaged.model_copy(update={"text": SOURCE + " For a constant speed v, distance is d = vt and time is t = d/v."})
    assert quality_issues(good) == []
    # Wikipag often drops pronunciation transcriptions; this is not a broken rule.
    assert quality_issues(good.model_copy(update={"text": "ASCII ( ), a character encoding. " + SOURCE})) == []


def test_no_usable_source_rejects_before_any_model_call(tmp_path):
    source = group().model_dump()
    source["passages"][0]["text"] += " It has a time requirement of , using big O notation."
    client = FakeClient()
    summary = asyncio.run(ApplicationCardBuilder(client, tmp_path).run([PassageGroup.model_validate(source)]))
    assert summary["complete"] and summary["cards"] == 0 and summary["rejected_groups"] == 1
    assert client.calls == []
    assert read_jsonl(tmp_path / "rejected_records.jsonl")[0]["rejection_stage"] == "source_preparation"


def local_corpus(tmp_path):
    corpus = tmp_path / "source"
    corpus.mkdir()
    shard = corpus / "wiki.jsonl"
    db = tmp_path / "offsets.sqlite"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE files (idx INTEGER PRIMARY KEY, path TEXT)")
    connection.execute("CREATE TABLE pos (id TEXT PRIMARY KEY, f INTEGER, off INTEGER) WITHOUT ROWID")
    connection.execute("INSERT INTO files VALUES (0, 'wiki.jsonl')")
    texts = [SOURCE, SOURCE + " A cost ) appears in a damaged sentence.",
             SOURCE + " Use this ratio only when the elapsed time is positive; divide the total distance by that time."]
    with shard.open("wb") as stream:
        for i, text in enumerate(texts):
            identity = f"12#{i}"
            connection.execute("INSERT INTO pos VALUES (?, 0, ?)", (identity, stream.tell()))
            stream.write((json.dumps({"id": identity, "title": "Speed", "text": text}) + "\n").encode())
    connection.commit()
    connection.close()
    preparer = SourcePreparer(offsets_db=db, corpus_root=corpus, source_id_prefix="wiki:")
    seed = PassageGroup(group_id="local", subject="physics", corpus="English Wikipag", language="en",
                        passages=[Passage(source_id="wiki:12#0", passage_id="12#0", title="Speed", text=SOURCE)])
    return preparer, seed, shard, db


def test_same_article_retrieval_is_read_only_cached_and_filters_damage(tmp_path, monkeypatch):
    preparer, seed, shard, db = local_corpus(tmp_path)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (shard, db)}
    output = tmp_path / "run"
    prepared, summary = preparer.prepare(seed, output)
    assert [p.passage_id for p in prepared.passages] == ["12#0", "12#2"]
    assert summary["supplementary_passages"] == 1 and summary["filtered_candidates"] == 1
    def forbidden(*args):
        raise AssertionError("resume must consume retrieval cache")
    monkeypatch.setattr(preparer, "_read", forbidden)
    resumed, repeated = preparer.prepare(seed, output)
    assert prepared == resumed and summary == repeated
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (shard, db)}
    cached = next((output / "cache" / "retrieval").glob("*.json"))
    saved = json.loads(cached.read_text())
    saved["payload"]["candidates"][0]["passage"]["text"] = "changed"
    cached.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="checksum"):
        preparer.prepare(seed, output)


def test_seed_must_match_actual_corpus_and_prefix(tmp_path):
    preparer, seed, _, _ = local_corpus(tmp_path)
    bad = seed.model_dump()
    bad["passages"][0]["text"] += " An invented statement."
    with pytest.raises(ValueError, match="differs from raw corpus"):
        preparer.prepare(PassageGroup.model_validate(bad), tmp_path / "run")
    bad["passages"][0]["source_id"] = "unrelated"
    with pytest.raises(ValueError, match="prefix"):
        preparer.prepare(PassageGroup.model_validate(bad), tmp_path / "run")


def test_post_review_rejection_never_exports_candidate(tmp_path):
    payloads = responses()
    payloads["card_review"]["cards"][0]["source_supported"] = False
    payloads["card_review"]["cards"][0]["issues"] = ["The quoted sentence does not support the claim."]
    client = FakeClient(payloads)
    result = asyncio.run(ApplicationCardBuilder(client, tmp_path, max_card_repairs=0).run([group()]))
    assert result["complete"] and result["rejected_groups"] == 1 and result["cards"] == 0
    assert read_jsonl(tmp_path / "card_rejections.jsonl")[0]["judgment"]["source_supported"] is False


@pytest.mark.parametrize("defect", ["missing", "duplicate", "out_of_range", "bad_quote"])
def test_review_coverage_and_evidence_checked(tmp_path, defect):
    payloads = responses()
    review = payloads["card_review"]["cards"]
    if defect == "missing":
        review.clear()
    elif defect == "duplicate":
        review.append(deepcopy(review[0]))
    elif defect == "out_of_range":
        review[0]["card_index"] = 1
    else:
        review[0]["evidence"][0]["quote"] = "This sentence does not exist."
    result = asyncio.run(ApplicationCardBuilder(FakeClient(payloads), tmp_path).run([group()]))
    assert not result["complete"] and result["cards"] == 0


def repair_payloads():
    payloads = responses()
    payloads["repair_review"] = deepcopy(payloads["card_review"])
    payloads["card_review"]["cards"][0]["conditions_preserved"] = False
    payloads["card_review"]["cards"][0]["issues"] = ["Require a positive elapsed time."]
    revised = deepcopy(payloads["extraction"]["cards"][0])
    revised["card"]["trigger"][0] += " Elapsed time is positive."
    payloads["card_repair"] = {"repairs": [{"card_index": 0, "replacement": revised,
                                             "reason": "Make the domain condition explicit."}]}
    return payloads


def test_bounded_repair_keeps_original_and_requires_fresh_review(tmp_path):
    payloads = repair_payloads()
    first = FakeClient(payloads, fail_stage="repair_review")
    assert not asyncio.run(ApplicationCardBuilder(first, tmp_path).run([group()]))["complete"]
    assert read_jsonl(tmp_path / "cards.jsonl") == []
    second = FakeClient(payloads)
    summary = asyncio.run(ApplicationCardBuilder(second, tmp_path).run([group()]))
    assert summary["cards"] == 1 and summary["groups_with_repair"] == 1
    assert [stage for stage, _ in second.calls] == ["repair_review"]
    assert "card_review" not in second.calls[0][1]["construction_record"]
    assert "positive" in read_jsonl(tmp_path / "cards.jsonl")[0]["trigger"][0]
    record = read_jsonl(tmp_path / "synthesis_records.jsonl")[0]
    assert "positive" not in record["records"]["extraction"]["cards"][0]["card"]["trigger"][0]
    third = FakeClient(payloads)
    hashes = {name: (tmp_path / name).read_bytes() for name in summary["output_sha256"]}
    asyncio.run(ApplicationCardBuilder(third, tmp_path).run([group()]))
    assert third.calls == []
    assert hashes == {name: (tmp_path / name).read_bytes() for name in hashes}


def test_failed_second_review_stops_without_a_repair_loop(tmp_path):
    payloads = repair_payloads()
    payloads["repair_review"] = deepcopy(payloads["card_review"])
    client = FakeClient(payloads)
    summary = asyncio.run(ApplicationCardBuilder(client, tmp_path).run([group()]))
    assert summary["complete"] and summary["cards"] == 0 and len(client.calls) == 7
    assert summary["rejected_card_reviews"] == 2


@pytest.mark.parametrize("defect", ["wrong_index", "unsupported", "empty"])
def test_repair_cannot_bypass_rejected_candidate_or_grounding(tmp_path, defect):
    payloads = repair_payloads()
    repairs = payloads["card_repair"]["repairs"]
    if defect == "wrong_index":
        repairs[0]["card_index"] = 2
    elif defect == "unsupported":
        repairs[0]["replacement"]["grounding"][0]["evidence"][0]["quote"] = "Made up source."
    else:
        repairs.clear()
    summary = asyncio.run(ApplicationCardBuilder(FakeClient(payloads), tmp_path).run([group()]))
    assert not summary["complete"] and summary["cards"] == 0


def test_null_repair_drops_card_without_export(tmp_path):
    payloads = repair_payloads()
    payloads["card_repair"]["repairs"][0]["replacement"] = None
    client = FakeClient(payloads)
    summary = asyncio.run(ApplicationCardBuilder(client, tmp_path).run([group()]))
    assert summary["complete"] and summary["cards"] == 0 and len(client.calls) == 6


def test_empty_extraction_is_explicit_rejection(tmp_path):
    payloads = responses()
    payloads["extraction"]["cards"] = []
    client = FakeClient(payloads)
    summary = asyncio.run(ApplicationCardBuilder(client, tmp_path).run([group()]))
    assert summary["complete"] and summary["rejected_groups"] == 1
    assert len(client.calls) == 4


def test_partial_acceptance_exports_only_reviewed_candidates(tmp_path):
    payloads = responses()
    second = deepcopy(payloads["extraction"]["cards"][0])
    second["card"]["concept_name"] = "Other speed"
    payloads["extraction"]["cards"].append(second)
    failed = deepcopy(payloads["card_review"]["cards"][0])
    failed.update(card_index=1, useful_application=False, issues=["Only repeats a definition."])
    payloads["card_review"]["cards"].append(failed)
    summary = asyncio.run(ApplicationCardBuilder(FakeClient(payloads), tmp_path, max_card_repairs=0).run([group()]))
    assert summary["complete"] and summary["cards"] == 1 and summary["rejected_card_reviews"] == 1
    assert read_jsonl(tmp_path / "cards.jsonl")[0]["concept_name"] == "Average speed"


def test_dropped_repair_preserves_other_candidate_and_original_index(tmp_path):
    payloads = repair_payloads()
    retained = deepcopy(payloads["extraction"]["cards"][0])
    retained["card"]["concept_name"] = "Retained speed rule"
    payloads["extraction"]["cards"].append(retained)
    good = deepcopy(payloads["repair_review"]["cards"][0])
    good["card_index"] = 1
    payloads["card_review"]["cards"].append(good)
    payloads["card_repair"]["repairs"][0]["replacement"] = None
    client = FakeClient(payloads)
    summary = asyncio.run(ApplicationCardBuilder(client, tmp_path).run([group()]))
    assert summary["complete"] and summary["cards"] == 1
    assert read_jsonl(tmp_path / "cards.jsonl")[0] == retained["card"]
    origin = read_jsonl(tmp_path / "bank_metadata.jsonl")[0]["synthesis_records"][0]
    assert origin["candidate_index"] == 0 and origin["original_candidate_index"] == 1
    assert origin["review_stage"] == "repair_review"
    assert client.calls[-1][1]["construction_record"]["extraction"]["cards"] == [retained]
