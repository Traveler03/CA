from __future__ import annotations

import json
from pathlib import Path

from src.methods.rag_baselines.pipeline import (
    build_evidence_context,
    load_existing_predictions,
    normalize_scores,
    normalize_text_list,
    parse_answer,
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
