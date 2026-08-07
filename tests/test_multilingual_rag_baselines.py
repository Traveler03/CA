from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_multilingual_rag_baselines import estimate_model_calls, main
from src.baselines.rag_methods import (
    ProcessedPassage,
    WikiPassage,
    qtt_passed,
    solver_messages,
    translate_query_messages,
)


def sample_row(language: str = "bn") -> dict[str, object]:
    return {
        "language": language,
        "subject": "abstract_algebra",
        "question": "প্রশ্ন",
        "options": {"A": "এক", "B": "দুই", "C": "তিন", "D": "চার"},
        "answer": "A",
        "sample_id": "demo/1",
    }


def test_estimated_calls_match_method_stages() -> None:
    rows = [sample_row("bn"), sample_row("hi")]
    assert estimate_model_calls(rows, ["trag", "dkm_rag", "qtt_rag"], generation_top_k=2) == {
        "trag": 4,
        "dkm_rag": 6,
        "qtt_rag": 6,
        "total": 16,
    }


def test_query_translation_prompt_preserves_labels() -> None:
    messages = translate_query_messages(sample_row("hi"))
    joined = "\n".join(message["content"] for message in messages)
    assert "Hindi" in joined
    assert '{"question":"...","options":{"A":"..."}}' in joined
    assert "A, B, C, D" in joined


def test_qtt_quality_threshold() -> None:
    assert qtt_passed({"semantic_equivalence": 3.5, "grammatical_accuracy": 4, "naturalness_fluency": 5})
    assert not qtt_passed({"semantic_equivalence": 3.4, "grammatical_accuracy": 5, "naturalness_fluency": 5})
    assert qtt_passed({"passed": True, "semantic_consistency": 0})


def test_processed_context_shapes() -> None:
    passage = WikiPassage(passage_id="p1", title="Title", text="English passage", language="en")
    dkm = ProcessedPassage(passage=passage, translated_text="অনুবাদ", refined_text="পরিমার্জিত")
    assert "Translated passage:\nঅনুবাদ" in dkm.context_text(method="dkm_rag")
    assert "Refined passage:\nপরিমার্জিত" in dkm.context_text(method="dkm_rag")
    qtt = ProcessedPassage(
        passage=passage,
        translated_text="অনুবাদ",
        quality_scores={"semantic_equivalence": 4, "grammatical_accuracy": 5, "naturalness_fluency": 4},
        outcome="translated_scored",
    )
    assert "semantic_equivalence=4" in qtt.context_text(method="qtt_rag")
    assert qtt.context_text(method="qtt_rag").endswith("অনুবাদ")


def test_solver_prompt_identifies_method() -> None:
    messages = solver_messages(sample_row("sw"), method="trag", retrieval_query="translated query", contexts=["doc"])
    joined = "\n".join(message["content"] for message in messages)
    assert "Method: trag" in joined
    assert "translated to English for retrieval" in joined
    assert '{"answer":"A"}' in joined


def test_mock_smoke_runner_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps(sample_row("bn"), ensure_ascii=False) + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    code = main(
        [
            "--input-jsonl",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--methods",
            "trag",
            "dkm_rag",
            "qtt_rag",
            "--retrieval-mode",
            "mock",
            "--max-rows",
            "1",
            "--candidate-top-k",
            "3",
            "--final-top-k",
            "1",
        ]
    )
    assert code == 0
    predictions = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["method"] for row in predictions] == ["trag", "dkm_rag", "qtt_rag"]
    assert predictions[0]["retrieval_scope"] == "english_only"
    assert predictions[1]["retrieval_scope"] == "multilingual"
    assert predictions[1]["translated_query"] is None
    assert predictions[2]["translated_query"] is None
    assert predictions[2]["processed_context"][0]["outcome"] == "original_query_language"
    assert (output_dir / "summary.json").exists()
