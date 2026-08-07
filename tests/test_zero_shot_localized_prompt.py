from __future__ import annotations

import pytest

from scripts.run_smoke_evaluation import (
    LOW_RESOURCE_ZERO_SHOT_PROMPTS,
    low_resource_zero_shot_prompt_parts,
    zero_shot_answer_repair_prompt,
    zero_shot_prompt,
)


def row_for(language: str) -> dict[str, object]:
    return {
        "language": language,
        "subject": "demo_subject",
        "question": "原始低资源语言题干",
        "options": {"A": "选项一", "B": "选项二", "C": "选项三", "D": "选项四"},
    }


def test_zero_shot_prompt_uses_language_specific_instruction_text() -> None:
    forbidden_english_fragments = [
        "You answer multiple-choice questions",
        "Return only JSON",
        "Question:",
        "Options:",
        "Subject:",
    ]

    for language, prompt_parts in LOW_RESOURCE_ZERO_SHOT_PROMPTS.items():
        messages = zero_shot_prompt(row_for(language))
        joined = "\n".join(message["content"] for message in messages)

        assert messages[0]["content"] == prompt_parts["system"]
        assert prompt_parts["question"] in joined
        assert prompt_parts["options"] in joined
        assert "原始低资源语言题干" in joined
        assert '{"answer":"A"}' in joined
        for fragment in forbidden_english_fragments:
            assert fragment not in joined


def test_zero_shot_repair_prompt_is_localized() -> None:
    messages = zero_shot_answer_repair_prompt(row_for("hi"), "previous response")
    joined = "\n".join(message["content"] for message in messages)

    assert "आप बहुविकल्पीय उत्तर" in messages[0]["content"]
    assert "पिछला मॉडल उत्तर" in joined
    assert "Extract the final intended answer" not in joined
    assert "Return only JSON" not in joined
    assert '{"answer":"A"}' in joined


def test_zero_shot_prompt_rejects_unsupported_language() -> None:
    with pytest.raises(ValueError, match="supported low-resource language"):
        low_resource_zero_shot_prompt_parts(row_for("en"))
