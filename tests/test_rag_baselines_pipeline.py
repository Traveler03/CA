from __future__ import annotations

import json
from pathlib import Path

from src.methods.rag_baselines.pipeline import (
    LLMResult,
    build_evidence_context,
    load_existing_predictions,
    normalize_scores,
    normalize_text_list,
    parse_answer,
    reset_method_outputs,
    run_qtt_rag,
)


class TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len((text or "").split())))

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"tok{i}" for i in ids)

    def token_count(self, text: str) -> int:
        return len((text or "").split())

    def truncate_tokens(self, text: str, max_tokens: int) -> str:
        return " ".join((text or "").split()[:max_tokens])


def test_parse_answer_supports_a_to_j() -> None:
    assert parse_answer('{"answer":"J"}', list("ABCDEFGHIJ")) == ("J", "json")
    assert parse_answer("Final answer: F", list("ABCDEFGHIJ")) == ("F", "regex")
    assert parse_answer("The answer is Z", list("ABCD")) == (None, "unparsed")


def test_normalize_text_list_preserves_length_and_fallback() -> None:
    assert normalize_text_list(["x", ""], 3, ["a", "b", "c"]) == ["x", "b", "c"]


def test_normalize_scores_clamps_and_fills() -> None:
    scores = normalize_scores([{"semantic_equivalence": 9, "grammatical_accuracy": "3"}], 2)
    assert scores == [
        {"semantic_equivalence": 5, "grammatical_accuracy": 3, "naturalness_fluency": 0},
        {"semantic_equivalence": 0, "grammatical_accuracy": 0, "naturalness_fluency": 0},
    ]


def test_dkm_evidence_budget_counts_refined_text() -> None:
    docs = [
        {
            "rank": 1,
            "title": "T",
            "section": "S",
            "language": "bn",
            "translated_text": "a " * 20,
            "refined_text": "b " * 20,
        }
    ]
    context, tokens = build_evidence_context(
        method="dkm_rag",
        docs=docs,
        tokenizer=TinyTokenizer(),  # type: ignore[arg-type]
        max_evidence_tokens=15,
        per_doc_max_tokens=15,
    )
    assert tokens <= 15
    assert context


def test_load_existing_predictions_last_writer_wins(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"eval_key": "x", "answer": "A"}),
                json.dumps({"eval_key": "x", "answer": "B"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_existing_predictions(path)["x"]["answer"] == "B"


def test_reset_method_outputs_clears_only_selected_method(tmp_path: Path) -> None:
    for subdir in ["retrieval", "translations", "dkm_refined", "qtt_tags", "predictions", "token_logs"]:
        target = tmp_path / subdir / "qtt_rag.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n", encoding="utf-8")
        other = tmp_path / subdir / "trag.jsonl"
        other.write_text("{}\n", encoding="utf-8")

    reset_method_outputs(tmp_path, "qtt_rag")

    for subdir in ["retrieval", "translations", "dkm_refined", "qtt_tags", "predictions", "token_logs"]:
        assert not (tmp_path / subdir / "qtt_rag.jsonl").exists()
        assert (tmp_path / subdir / "trag.jsonl").exists()


class DummyLLM:
    def __init__(self) -> None:
        self.stage_counts: dict[str, int] = {}

    def token_count(self, text: str) -> int:
        return len((text or "").split())

    def truncate_tokens(self, text: str, max_tokens: int) -> str:
        return " ".join((text or "").split()[:max_tokens])

    def generate_many(self, *, stage: str, messages_list: list[list[dict[str, str]]], max_tokens: int) -> list[LLMResult]:
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + len(messages_list)
        if stage == "qtt_translate_non_query_passages":
            raise AssertionError("QTT should not translate same-language retrieved passages")
        if stage == "qtt_score_quality":
            content = '{"scores":[{"semantic_equivalence":5,"grammatical_accuracy":5,"naturalness_fluency":5}]}'
        elif stage == "qtt_answer":
            content = "A"
        else:
            content = "{}"
        return [
            LLMResult(
                content=content,
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
                latency_ms=3,
                finish_reason="stop",
                cache_hit=False,
            )
            for _ in messages_list
        ]


class SameLanguageRetriever:
    def retrieve(self, query: str, top_k: int) -> dict[str, object]:
        return {
            "query": query,
            "latency_ms": 7,
            "results": [
                {
                    "rank": 1,
                    "score": 1.0,
                    "language": "bn",
                    "passage_id": "bn_1",
                    "title": "Title",
                    "section": "Lead",
                    "text": "same language evidence",
                }
            ],
        }


def test_qtt_skips_noop_translation_for_same_language_docs(tmp_path: Path) -> None:
    row = {
        "_dataset": "unit",
        "_language": "bn",
        "eval_key": "unit|bn|sample:1",
        "question": "প্রশ্ন?",
        "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
        "answer": "A",
        "sample_id": "1",
        "subject": "unit_subject",
    }
    config = {
        "resume": False,
        "rag": {
            "top_k": 1,
            "translation_max_tokens": 64,
            "quality_score_max_tokens": 64,
            "max_final_answer_tokens": 8,
            "max_evidence_tokens": 1200,
            "per_doc_max_tokens": 240,
        },
    }
    llm = DummyLLM()

    run_qtt_rag(
        rows=[row],
        config=config,
        output_dir=tmp_path,
        llm=llm,  # type: ignore[arg-type]
        multilingual_retriever=SameLanguageRetriever(),  # type: ignore[arg-type]
    )

    token_log = json.loads((tmp_path / "token_logs" / "qtt_rag.jsonl").read_text(encoding="utf-8"))
    assert token_log["num_llm_calls"] == 2
    assert "translate_non_query_passages" in token_log["stage_level_token_counts"]
    assert token_log["stage_level_token_counts"]["translate_non_query_passages"]["total_tokens"] == 0
    assert llm.stage_counts == {"qtt_score_quality": 1, "qtt_answer": 1}
