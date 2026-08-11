from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import Any

from scripts.run_multilingual_rag_baselines import JsonCache, MockModel, RetrievalResources, main, parse_answer_label, run_one
from src.baselines.coral_wikipag import (
    CoralCard,
    CoralScoredCard,
    MockCoralUsageBank,
    accumulate_coral_cards,
    coral_card_passes,
    coral_score_total,
    run_coral_wikipag,
)


def sample_row() -> dict[str, Any]:
    return {
        "language": "bn",
        "subject": "abstract_algebra",
        "question": "প্রশ্ন",
        "options": {"A": "এক", "B": "দুই", "C": "তিন", "D": "চার"},
        "answer": "A",
        "sample_id": "demo/1",
        "_dataset": "global_mmlu",
    }


def card(language: str = "bn", document_id: str = "doc", score: float = 0.9) -> CoralCard:
    return CoralCard(
        language=language,
        document_id=document_id,
        subject="abstract_algebra",
        concept_id="C1",
        usage_id=document_id,
        index_key="key",
        payload="payload",
        text="card text",
        score=score,
        rank=1,
    )


class FakeModel:
    def __init__(self, *, enough_after: int = 1) -> None:
        self.calls = 0
        self.enough_after = enough_after

    async def json(
        self,
        messages: list[dict[str, str]],
        *,
        namespace: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        self.calls += 1
        system = messages[0]["content"].lower()
        if "planner" in system:
            if ".round1." in namespace:
                return {"language_names": ["bn"]}
            return {"language_names": ["bn"], "rewritten_query": "rewritten query"}
        if "translate multilingual multiple-choice" in system:
            return {"question": "translated question", "options": {"A": "one", "B": "two", "C": "three", "D": "four"}}
        if "batch card critic" in system:
            user = messages[-1]["content"]
            count = max(1, user.count('"concept_id"'))
            return {
                "cards": [
                    {
                        "id": idx,
                        "scores": {
                            "relevance": 4,
                            "usefulness": 4,
                            "clarity_specificity": 4,
                            "compatibility": 4,
                        },
                        "critique": "useful",
                    }
                    for idx in range(count)
                ]
            }
        if "card critic" in system:
            return {
                "scores": {
                    "relevance": 4,
                    "usefulness": 4,
                    "clarity_specificity": 4,
                    "compatibility": 4,
                },
                "critique": "useful",
            }
        if "sufficiency critic" in system:
            round_no = 1 if ".round1." in namespace else 2
            return {"enough_documents": round_no >= self.enough_after, "reason": "need more" if round_no < self.enough_after else "enough"}
        return {"answer": "A"}


class EmptyBank(MockCoralUsageBank):
    def retrieve(self, *, subject: str, query: str, language: str, top_k: int) -> list[CoralCard]:
        return []

    def triples_for_concepts(self, concept_ids: list[str], *, max_triples: int) -> list[dict[str, Any]]:
        return []


def test_coral_score_formula_and_thresholds() -> None:
    scores = {"relevance": 3, "usefulness": 4, "clarity_specificity": 2, "compatibility": 5}
    assert coral_score_total(scores) == 8.5
    kept, reason, total = coral_card_passes(scores, min_each=2, min_total=6)
    assert kept
    assert reason == "passed"
    assert total == 8.5
    kept, reason, _ = coral_card_passes({"relevance": 5, "usefulness": 5, "clarity_specificity": 1, "compatibility": 5})
    assert not kept
    assert "clarity_specificity" in reason


def test_coral_accumulate_dedup_keeps_highest_score() -> None:
    low = CoralScoredCard(card=card(score=0.7), scores={}, critique="", s_tot=6.0, kept=True, keep_reason="passed", round_idx=1)
    high = CoralScoredCard(card=card(score=0.8), scores={}, critique="", s_tot=8.0, kept=True, keep_reason="passed", round_idx=2)
    rejected = CoralScoredCard(
        card=card(document_id="reject"), scores={}, critique="", s_tot=10.0, kept=False, keep_reason="rejected", round_idx=2
    )
    out = accumulate_coral_cards({}, [low])
    out = accumulate_coral_cards(out, [high, rejected])
    assert len(out) == 1
    assert out[("bn", "doc")].s_tot == 8.0


def test_coral_run_stops_early_when_sufficient() -> None:
    result = asyncio.run(run_coral_wikipag(
        sample_row(),
        model=FakeModel(enough_after=1),
        bank=MockCoralUsageBank(),
        config={"coral_max_rounds": 3, "coral_max_corpora": 1, "coral_top_k_per_corpus": 2},
        row_key="coral_wikipag|global_mmlu|bn|demo/1",
        parse_answer_label=parse_answer_label,
    ))
    assert result["stop_reason"] == "sufficient"
    assert len(result["trace"]["rounds"]) == 1
    assert result["correct"]


def test_coral_run_rewrites_query_and_stops_at_max_rounds() -> None:
    result = asyncio.run(run_coral_wikipag(
        sample_row(),
        model=FakeModel(enough_after=99),
        bank=MockCoralUsageBank(),
        config={"coral_max_rounds": 2, "coral_max_corpora": 1, "coral_top_k_per_corpus": 1},
        row_key="coral_wikipag|global_mmlu|bn|demo/1",
        parse_answer_label=parse_answer_label,
    ))
    assert result["stop_reason"] == "max_rounds"
    assert result["final_query"] == "rewritten query"
    assert len(result["trace"]["rounds"]) == 2


def test_coral_empty_evidence_falls_back_to_unified_answer_model() -> None:
    result = asyncio.run(run_coral_wikipag(
        sample_row(),
        model=FakeModel(enough_after=99),
        bank=EmptyBank(),
        config={"coral_max_rounds": 1, "coral_max_corpora": 1, "coral_top_k_per_corpus": 2},
        row_key="coral_wikipag|global_mmlu|bn|demo/1",
        parse_answer_label=parse_answer_label,
    ))
    assert result["fallback_reason"] == "no_qualified_coral_wikipag_cards"
    assert result["final_cards"] == []
    assert result["correct"]


def test_coral_batch_critic_and_translated_query() -> None:
    result = asyncio.run(run_coral_wikipag(
        sample_row(),
        model=FakeModel(enough_after=1),
        bank=MockCoralUsageBank(),
        config={
            "coral_max_rounds": 1,
            "coral_max_corpora": 1,
            "coral_top_k_per_corpus": 3,
            "coral_batch_critic": True,
            "coral_translate_query": True,
        },
        row_key="coral_wikipag|global_mmlu|bn|demo/1",
        parse_answer_label=parse_answer_label,
    ))
    assert result["translated_query"]["question"] == "translated question"
    assert "translated question" in result["final_query"]
    assert len(result["final_cards"]) == 3
    assert result["coral_config"]["batch_critic"]


def test_coral_cli_mock_resume_keeps_single_prediction(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps(sample_row(), ensure_ascii=False) + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    argv = [
        "--input-jsonl",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--methods",
        "coral_wikipag",
        "--retrieval-mode",
        "mock",
        "--max-rows",
        "1",
        "--max-model-calls",
        "20",
        "--coral-max-rounds",
        "1",
        "--coral-max-corpora",
        "1",
        "--coral-top-k-per-corpus",
        "2",
        "--coral-batch-critic",
        "--resume",
    ]
    assert main(argv) == 0
    assert main(argv) == 0
    rows = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["method"] == "coral_wikipag"
    assert rows[0]["method_name"] == "CORAL-Wikipag"


def test_coral_drag_fallback_runs_when_gate_selects_coral(tmp_path: Path) -> None:
    result = asyncio.run(run_one(
        sample_row(),
        "coral_wikipag",
        model=MockModel(),
        config={
            "retrieval_mode": "mock",
            "coral_max_rounds": 1,
            "coral_max_corpora": 1,
            "coral_top_k_per_corpus": 2,
            "coral_batch_critic": True,
            "coral_adaptive_direct_gate": True,
            "coral_drag_fallback_on_card_gate": True,
            "coral_drag_fallback_reranker": "none",
            "final_top_k": 2,
            "answer_max_tokens": 64,
            "drag_icl_answer_max_tokens": 64,
        },
        resources=RetrievalResources(
            cache=JsonCache(tmp_path / "cache"),
            coral_bank=MockCoralUsageBank(),
        ),
    ))
    assert result["method"] == "coral_wikipag"
    assert result["gate_selected_source"] == "coral"
    assert result["coral_drag_fallback_enabled"]
    assert result["coral_drag_fallback_applied"]
    assert result["coral_drag_fallback"]["method"] == "drag_icl"
    assert result["coral_drag_fallback"]["reranker"] == "none"
    assert result["prediction"] == "A"


def test_coral_drag_fallback_policy_has_cards(tmp_path: Path) -> None:
    result = asyncio.run(run_one(
        sample_row(),
        "coral_wikipag",
        model=MockModel(),
        config={
            "retrieval_mode": "mock",
            "coral_max_rounds": 1,
            "coral_max_corpora": 1,
            "coral_top_k_per_corpus": 2,
            "coral_batch_critic": True,
            "coral_adaptive_direct_gate": True,
            "coral_drag_fallback_on_card_gate": True,
            "coral_drag_fallback_policy": "has_cards",
            "coral_drag_fallback_reranker": "none",
            "final_top_k": 2,
            "answer_max_tokens": 64,
            "drag_icl_answer_max_tokens": 64,
        },
        resources=RetrievalResources(
            cache=JsonCache(tmp_path / "cache"),
            coral_bank=MockCoralUsageBank(),
        ),
    ))
    assert result["coral_drag_fallback_policy"] == "has_cards"
    assert result["coral_drag_fallback_applied"]
    assert result["coral_drag_fallback_reason"] == "has_cards"
    assert result["coral_drag_fallback"]["method"] == "drag_icl"
