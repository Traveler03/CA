from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal


RagMethod = Literal["trag", "dkm_rag", "qtt_rag", "multirag", "drag_icl", "coral_wikipag"]


LANGUAGE_NAMES = {
    "bn": "Bengali",
    "hi": "Hindi",
    "sw": "Swahili",
    "te": "Telugu",
    "ne": "Nepali",
    "en": "English",
}


@dataclass(frozen=True)
class WikiPassage:
    passage_id: str
    title: str
    text: str
    score: float = 0.0
    section: str | None = None
    language: str = "en"
    rank: int | None = None


@dataclass(frozen=True)
class ProcessedPassage:
    passage: WikiPassage
    translated_text: str | None = None
    refined_text: str | None = None
    keep: bool = True
    quality_scores: dict[str, float] = field(default_factory=dict)
    outcome: str = "kept"

    def context_text(self, *, method: RagMethod) -> str:
        section = f" / {self.passage.section}" if self.passage.section else ""
        base = f"[{self.passage.title}{section}] ({self.passage.language})"
        if method == "dkm_rag":
            translated = self.translated_text or self.passage.text
            refined = self.refined_text or ""
            return (
                f"{base}\n"
                f"Translated passage:\n{translated}\n"
                f"Refined passage:\n{refined}"
            ).strip()
        if method == "qtt_rag":
            if self.outcome == "original_query_language":
                return f"{base}\nOriginal query-language document:\n{self.passage.text}".strip()
            scores = self.quality_scores or {}
            return (
                f"{base}\n"
                "Translation quality tags: "
                f"semantic_equivalence={scores.get('semantic_equivalence', 0.0)}, "
                f"grammatical_accuracy={scores.get('grammatical_accuracy', 0.0)}, "
                f"naturalness_fluency={scores.get('naturalness_fluency', 0.0)}\n"
                f"Translated passage:\n{self.translated_text or self.passage.text}"
            ).strip()
        return f"{base}\n{self.passage.text}".strip()


def supported_language(row: dict[str, Any]) -> str:
    language = str(row.get("language") or row.get("_language") or "").strip()
    if language not in LANGUAGE_NAMES:
        raise ValueError(f"unsupported language {language!r}; expected one of {sorted(LANGUAGE_NAMES)}")
    return language


def option_labels(options: dict[str, Any]) -> list[str]:
    return sorted(str(label).upper() for label in options if re.fullmatch(r"[A-J]", str(label).upper()))


def format_options(options: dict[str, Any]) -> str:
    return "\n".join(f"{label}. {options[label]}" for label in option_labels(options))


def question_text(row: dict[str, Any]) -> str:
    return (
        f"Subject: {row.get('subject', '')}\n"
        f"Question:\n{row.get('question', '')}\n\n"
        f"Options:\n{format_options(row.get('options') or {})}"
    )


def retrieval_query_text(row: dict[str, Any]) -> str:
    return (
        f"{row.get('question', '')}\n\n"
        f"Options:\n{format_options(row.get('options') or {})}"
    ).strip()


def english_query_from_translated_payload(payload: dict[str, Any]) -> str:
    question = str(payload.get("question") or "").strip()
    options = payload.get("options") or {}
    if isinstance(options, dict):
        option_text = " ".join(f"{label}. {text}" for label, text in sorted(options.items()))
    else:
        option_text = str(options)
    return re.sub(r"\s+", " ", f"{question} {option_text}").strip()


def mock_translated_question(row: dict[str, Any]) -> dict[str, Any]:
    """Deterministic smoke translation placeholder.

    It preserves labels and structure so the full pipeline can be tested without
    downloading NLLB or calling a live LLM.
    """

    return {
        "question": str(row.get("question") or ""),
        "options": {label: str(text) for label, text in (row.get("options") or {}).items()},
    }


def parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model response")
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    answer_match = re.search(r'"answer"\s*:\s*"([A-J])"', text, flags=re.IGNORECASE)
    if answer_match:
        return {"answer": answer_match.group(1).upper(), "_parse_fallback": "answer_field_regex"}
    section_match = re.search(r"(?:#\s*Answer|Answer)\s*[:\n\r\s-]*([A-J])\b", text, flags=re.IGNORECASE)
    if section_match:
        return {"answer": section_match.group(1).upper(), "_parse_fallback": "answer_section_regex"}
    raise ValueError(f"no JSON object found: {text[:200]}")


def translate_query_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    language = supported_language(row)
    language_name = LANGUAGE_NAMES[language]
    labels = ", ".join(option_labels(row.get("options") or {}))
    return [
        {
            "role": "system",
            "content": (
                "You translate multilingual multiple-choice exam queries into English for retrieval. "
                "Preserve option labels exactly. Output only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source language: {language_name} ({language})\n"
                f"{question_text(row)}\n\n"
                "Translate the question and options into English. "
                f"Return exactly this JSON schema with option labels {labels}: "
                '{"question":"...","options":{"A":"..."}}'
            ),
        },
    ]


def translate_passage_messages(passage: WikiPassage, *, target_language: str) -> list[dict[str, str]]:
    source_language_name = LANGUAGE_NAMES.get(passage.language, passage.language)
    target_language_name = LANGUAGE_NAMES[target_language]
    return [
        {
            "role": "system",
            "content": (
                "You are a faithful document translator for multilingual RAG. "
                "Preserve technical terms, numbers, named entities, and equations. Output only the translation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source language: {source_language_name} ({passage.language})\n"
                f"Target language: {target_language_name} ({target_language})\n\n"
                f"Translate the following Wikipedia passage into {target_language_name}:\n\n{passage.text}"
            ),
        },
    ]


def translate_passages_batch_messages(passages: list[WikiPassage], *, target_language: str) -> list[dict[str, str]]:
    target_language_name = LANGUAGE_NAMES[target_language]
    compact = [
        {
            "id": idx,
            "source_language": passage.language,
            "title": passage.title,
            "section": passage.section,
            "text": passage.text[:500],
        }
        for idx, passage in enumerate(passages)
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are a faithful document translator for multilingual RAG. "
                "Preserve technical terms, numbers, named entities, and equations. Output only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Translate each Wikipedia passage into {target_language_name} ({target_language}). "
                'Return exactly JSON: {"translations":["..."]}. Keep the same order.\n\n'
                + json.dumps(compact, ensure_ascii=False)
            ),
        },
    ]


def dkm_refine_messages(row: dict[str, Any], translated_passage: str) -> list[dict[str, str]]:
    language = supported_language(row)
    language_name = LANGUAGE_NAMES[language]
    return [
        {
            "role": "system",
            "content": (
                "Create a standalone document for Dual Knowledge Multilingual RAG. "
                "Use facts related to the question, remove repetition, and connect the content smoothly. "
                "Do not add an answer label. Output only the refined document."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Write in {language_name}.\n\n"
                f"Question:\n{row.get('question', '')}\n\n"
                f"Translated passage:\n{translated_passage}"
            ),
        },
    ]


def dkm_refine_batch_messages(row: dict[str, Any], translated_passages: list[str]) -> list[dict[str, str]]:
    language = supported_language(row)
    language_name = LANGUAGE_NAMES[language]
    compact = {
        "question": row.get("question", ""),
        "options": row.get("options") or {},
        "translated_passages": translated_passages,
    }
    return [
        {
            "role": "system",
            "content": (
                "Create concise standalone documents for Dual Knowledge Multilingual RAG. "
                "Use only facts related to the question, remove repetition, and connect the content smoothly. "
                "Do not add an answer label. Keep each refined document under 90 words. Output only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Write in {language_name} ({language}). For each translated passage, produce one refined document. "
                'Return exactly JSON: {"refined":["..."]}. Keep the same order.\n\n'
                + json.dumps(compact, ensure_ascii=False)
            ),
        },
    ]


def qtt_quality_messages(original: str, translated: str, *, target_language: str) -> list[dict[str, str]]:
    language_name = LANGUAGE_NAMES[target_language]
    return [
        {
            "role": "system",
            "content": (
                "Evaluate translation quality for Quality-Aware Translation Tagging. "
                "Score semantic_equivalence, grammatical_accuracy, and naturalness_fluency from 0.0 to 5.0. "
                "Do not filter or rewrite the evidence. Output only JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Target language: {language_name}\n\n"
                f"Original source passage:\n{original}\n\n"
                f"Translated passage:\n{translated}\n\n"
                'Return JSON: {"semantic_equivalence":4.0,"grammatical_accuracy":4.0,'
                '"naturalness_fluency":4.0}'
            ),
        },
    ]


def qtt_quality_batch_messages(pairs: list[dict[str, str]], *, target_language: str) -> list[dict[str, str]]:
    language_name = LANGUAGE_NAMES[target_language]
    compact = [
        {
            "id": idx,
            "original_source_passage": pair.get("original", "")[:500],
            "translated_passage": pair.get("translated", "")[:500],
        }
        for idx, pair in enumerate(pairs)
    ]
    return [
        {
            "role": "system",
            "content": (
                "Evaluate translation quality for Quality-Aware Translation Tagging. "
                "Score semantic_equivalence, grammatical_accuracy, and naturalness_fluency from 0.0 to 5.0. "
                "Do not filter or rewrite the evidence. Output only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Target language: {language_name} ({target_language}). "
                'Return exactly JSON: {"scores":[{"semantic_equivalence":4.0,'
                '"grammatical_accuracy":4.0,"naturalness_fluency":4.0}]}. Keep the same order.\n\n'
                + json.dumps(compact, ensure_ascii=False)
            ),
        },
    ]


def solver_messages(
    row: dict[str, Any],
    *,
    method: RagMethod,
    retrieval_query: str,
    contexts: list[str],
    translated_query: dict[str, Any] | None = None,
    retrieval_scope: str | None = None,
) -> list[dict[str, str]]:
    language = supported_language(row)
    language_name = LANGUAGE_NAMES[language]
    context_block = "\n\n".join(f"Document {idx + 1}:\n{text}" for idx, text in enumerate(contexts)) or "No retrieved context."
    labels = ", ".join(option_labels(row.get("options") or {}))
    method_note = {
        "trag": "The query was translated to English for retrieval; documents are retrieved English evidence.",
        "multirag": (
            "The original multilingual query was used for retrieval. Documents are original retrieved evidence; "
            "do not translate or rewrite them. Some retrieved documents may be irrelevant; ignore irrelevant evidence "
            "and do not let it override a clear answer from the question and options."
        ),
        "drag_icl": (
            "Use the D-RAG-ICL four-stage evidence procedure. Documents are original retrieved evidence; "
            "do not translate or rewrite them before reasoning."
        ),
        "coral_wikipag": (
            "Use CORAL-Wikipag verified usage cards and concept triples when available. "
            "If no verified cards are provided, answer from the question and options."
        ),
        "dkm_rag": "Documents contain translated passages plus refined passages. Use both; the refined passage is not an answer key.",
        "qtt_rag": (
            "Use original query-language documents first when present. For translated documents, consider all three "
            "quality tags: semantic_equivalence, grammatical_accuracy, and naturalness_fluency. Do not assume any "
            "retrieved document was filtered."
        ),
    }[method]
    translation_block = ""
    if translated_query:
        translation_block = "English translated query:\n" + json.dumps(translated_query, ensure_ascii=False) + "\n\n"
    return [
        {
            "role": "system",
            "content": (
                "You answer multilingual multiple-choice exam questions with retrieved evidence. "
                "Hidden reasoning is disabled. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Target language: {language_name} ({language})\n"
                f"Method: {method}\n"
                f"Retrieval scope: {retrieval_scope or 'unspecified'}\n"
                f"{method_note}\n"
                f"{translation_block}"
                f"Retrieval query:\n{retrieval_query}\n\n"
                f"Retrieved context:\n{context_block}\n\n"
                f"{question_text(row)}\n\n"
                f"Choose the single best answer from {labels}. "
                'Return only JSON: {"answer":"A"}'
            ),
        },
    ]


def drag_icl_solver_messages(
    row: dict[str, Any],
    *,
    retrieval_query: str,
    contexts: list[str],
    retrieval_scope: str | None = None,
) -> list[dict[str, str]]:
    language = supported_language(row)
    language_name = LANGUAGE_NAMES[language]
    context_block = "\n\n".join(f"Document {idx + 1}:\n{text}" for idx, text in enumerate(contexts)) or "No retrieved context."
    labels = ", ".join(option_labels(row.get("options") or {}))
    return [
        {
            "role": "system",
            "content": (
                "You run D-RAG-ICL adapted for multilingual multiple-choice QA. "
                "Use retrieved evidence when it is relevant; ignore irrelevant evidence. "
                "Complete the four requested stages and end with a single answer label."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Target language: {language_name} ({language})\n"
                f"Method: drag_icl\n"
                f"Retrieval scope: {retrieval_scope or 'multilingual'}\n"
                f"Retrieval query:\n{retrieval_query}\n\n"
                f"# Reference Evidence:\n{context_block}\n\n"
                f"# Question:\n{question_text(row)}\n\n"
                "Complete all four stages in one response:\n"
                "#Extraction\n"
                "In English, extract at most 3 short evidence points that could help answer the question.\n\n"
                "#Explaination\n"
                "In English, write at most 5 short document relevance notes. Cite Document numbers.\n\n"
                "#Dialectic Argumentation\n"
                "In English, compare evidence in at most 40 words. If evidence is weak or irrelevant, say so and use the question/options.\n\n"
                "#Answer\n"
                f"Choose the single best answer from {labels}. "
                "Write the final line exactly as: Answer: A"
            ),
        },
    ]


def adaptive_evidence_gate_messages(
    row: dict[str, Any],
    *,
    method: RagMethod,
    retrieval_query: str,
    contexts: list[str],
    zero_shot_answer: str | None,
    rag_answer: str | None,
) -> list[dict[str, str]]:
    language = supported_language(row)
    language_name = LANGUAGE_NAMES[language]
    context_block = "\n\n".join(f"Document {idx + 1}:\n{text}" for idx, text in enumerate(contexts)) or "No retrieved context."
    labels = ", ".join(option_labels(row.get("options") or {}))
    return [
        {
            "role": "system",
            "content": (
                "You are a conservative evidence gate for multilingual multiple-choice RAG. "
                "Hidden reasoning is disabled. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Target language: {language_name} ({language})\n"
                f"Method being gated: {method}\n"
                f"Retrieval query:\n{retrieval_query}\n\n"
                f"Retrieved evidence:\n{context_block}\n\n"
                f"{question_text(row)}\n\n"
                f"No-evidence answer: {zero_shot_answer or 'INVALID'}\n"
                f"RAG answer: {rag_answer or 'INVALID'}\n\n"
                "Choose between the no-evidence answer and the RAG answer only.\n"
                "Use the RAG answer only if at least one retrieved document directly supports that answer "
                "or directly eliminates the no-evidence answer. If the evidence is generic, off-topic, "
                "too weak, incomplete, or conflicting, choose the no-evidence answer. "
                f"The final answer must be one of {labels}. "
                'Return exactly JSON: {"answer":"A","selected_source":"zero_shot|rag",'
                '"evidence_status":"direct_support|weak_or_irrelevant|conflicting|invalid_candidate"}'
            ),
        },
    ]


def qtt_passed(scores: dict[str, Any], threshold: float = 3.5) -> bool:
    """Compatibility helper for ablations only.

    The faithful QTT-RAG path tags translated evidence and keeps all retrieved
    documents. It must not call this helper for main experiments.
    """

    if isinstance(scores.get("passed"), bool):
        return bool(scores["passed"])
    keys = ["semantic_equivalence", "grammatical_accuracy", "naturalness_fluency"]
    legacy_keys = ["semantic_consistency", "grammatical_accuracy", "fluency"]
    if not any(key in scores for key in keys):
        keys = legacy_keys
    try:
        return all(float(scores.get(key, 0.0)) >= threshold for key in keys)
    except Exception:
        return False


def normalize_quality_scores(scores: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    aliases = {
        "semantic_equivalence": ["semantic_equivalence", "semantic_consistency"],
        "grammatical_accuracy": ["grammatical_accuracy"],
        "naturalness_fluency": ["naturalness_fluency", "fluency"],
    }
    for key, raw_keys in aliases.items():
        value = 0.0
        for raw_key in raw_keys:
            if raw_key in scores:
                try:
                    value = float(scores.get(raw_key, 0.0))
                except Exception:
                    value = 0.0
                break
        out[key] = value
    return out
