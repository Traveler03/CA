from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_multilingual_rag_baselines import estimate_model_calls, main, parse_answer_label_from_text
from src.baselines.rag_methods import (
    ProcessedPassage,
    WikiPassage,
    adaptive_evidence_gate_messages,
    drag_icl_solver_messages,
    parse_json_object,
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
    assert estimate_model_calls(rows, ["trag", "dkm_rag", "qtt_rag", "multirag", "drag_icl"], generation_top_k=2) == {
        "trag": 4,
        "dkm_rag": 6,
        "qtt_rag": 6,
        "multirag": 2,
        "drag_icl": 2,
        "total": 20,
    }
    assert estimate_model_calls(
        rows,
        ["multirag", "drag_icl"],
        generation_top_k=2,
        adaptive_evidence_gate=True,
    ) == {
        "multirag": 6,
        "drag_icl": 6,
        "total": 12,
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


def test_parse_json_object_recovers_truncated_answer_field() -> None:
    assert parse_json_object('{"answer":"C","extraction":["long text"')["answer"] == "C"
    assert parse_json_object("#Answer\nD\nmore text")["answer"] == "D"


def test_parse_answer_label_from_drag_text() -> None:
    assert parse_answer_label_from_text("#Extraction\n...\n#Answer\nAnswer: B", sample_row("bn")) == ("B", True)


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
    assert '"answer":"A"' in joined


def test_multirag_prompt_uses_original_evidence() -> None:
    messages = solver_messages(sample_row("ne"), method="multirag", retrieval_query="original query", contexts=["doc"])
    joined = "\n".join(message["content"] for message in messages)
    assert "Method: multirag" in joined
    assert "original multilingual query" in joined
    assert "do not translate or rewrite" in joined


def test_drag_icl_prompt_has_four_required_sections() -> None:
    messages = drag_icl_solver_messages(sample_row("te"), retrieval_query="original query", contexts=["doc1", "doc2"])
    joined = "\n".join(message["content"] for message in messages)
    assert "#Extraction" in joined
    assert "#Explaination" in joined
    assert "#Dialectic Argumentation" in joined
    assert "#Answer" in joined
    assert "Answer: A" in joined


def test_adaptive_gate_prompt_is_conservative() -> None:
    messages = adaptive_evidence_gate_messages(
        sample_row("bn"),
        method="drag_icl",
        retrieval_query="original query",
        contexts=["doc1"],
        zero_shot_answer="B",
        rag_answer="C",
    )
    joined = "\n".join(message["content"] for message in messages)
    assert "Choose between the no-evidence answer and the RAG answer only" in joined
    assert "Use the RAG answer only if" in joined
    assert "No-evidence answer: B" in joined
    assert "RAG answer: C" in joined


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
            "multirag",
            "drag_icl",
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
    assert [row["method"] for row in predictions] == ["trag", "dkm_rag", "qtt_rag", "multirag", "drag_icl"]
    assert predictions[0]["retrieval_scope"] == "english_only"
    assert predictions[1]["retrieval_scope"] == "multilingual"
    assert predictions[1]["translated_query"] is None
    assert predictions[2]["translated_query"] is None
    assert predictions[2]["processed_context"][0]["outcome"] == "original_query_language"
    assert predictions[3]["retrieval_scope"] == "multilingual"
    assert predictions[4]["retrieval_scope"] == "multilingual"
    assert predictions[4]["candidate_top_k"] == 10
    assert predictions[4]["valid"]
    assert (output_dir / "summary.json").exists()


def test_mock_adaptive_gate_writes_trace_fields(tmp_path: Path) -> None:
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
            "drag_icl",
            "--retrieval-mode",
            "mock",
            "--max-rows",
            "1",
            "--final-top-k",
            "1",
            "--adaptive-evidence-gate",
            "--adaptive-min-rag-score",
            "0.6",
        ]
    )
    assert code == 0
    row = json.loads((output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["adaptive_evidence_gate"]
    assert row["rag_prediction"] == "A"
    assert row["zero_shot_prediction"] == "A"
    assert row["top_retrieval_score"] == 1.0
    assert row["score_gate_applied"] is False
