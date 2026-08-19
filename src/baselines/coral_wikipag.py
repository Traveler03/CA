from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.baselines.rag_methods import (
    LANGUAGE_NAMES,
    english_query_from_translated_payload,
    format_options,
    option_labels,
    supported_language,
    translate_query_messages,
)


DEFAULT_CORAL_BANK_DIR = Path(
    "/tmp/ca_deleted_old_methods_20260807_125106/"
    "runs/full_20260730/wikipag_clean_57subjects_target1000_gpt54_combined"
)
DEFAULT_LANGUAGE_POOL = ["bn", "sw", "te", "ne", "hi", "en"]


@dataclass(frozen=True)
class CoralCard:
    language: str
    document_id: str
    subject: str
    concept_id: str
    usage_id: str
    index_key: str
    payload: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoralScoredCard:
    card: CoralCard
    scores: dict[str, float]
    critique: str
    s_tot: float
    kept: bool
    keep_reason: str
    round_idx: int

    @property
    def dedupe_key(self) -> tuple[str, str]:
        return (self.card.language, self.card.document_id)


def coral_score_total(scores: dict[str, Any]) -> float:
    relevance = float(scores.get("relevance", 0.0) or 0.0)
    usefulness = float(scores.get("usefulness", 0.0) or 0.0)
    clarity = float(scores.get("clarity_specificity", 0.0) or 0.0)
    compatibility = float(scores.get("compatibility", 0.0) or 0.0)
    return relevance + 0.5 * (usefulness + clarity + compatibility)


def coral_card_passes(
    scores: dict[str, Any],
    *,
    min_each: float = 2.0,
    min_total: float = 6.0,
) -> tuple[bool, str, float]:
    normalized = {
        "relevance": float(scores.get("relevance", 0.0) or 0.0),
        "usefulness": float(scores.get("usefulness", 0.0) or 0.0),
        "clarity_specificity": float(scores.get("clarity_specificity", 0.0) or 0.0),
        "compatibility": float(scores.get("compatibility", 0.0) or 0.0),
    }
    s_tot = coral_score_total(normalized)
    low = [key for key, value in normalized.items() if value < min_each]
    if low:
        return False, f"score_below_min_each:{','.join(low)}", s_tot
    if s_tot < min_total:
        return False, f"total_below_threshold:{s_tot:.3f}<{min_total:.3f}", s_tot
    return True, "passed", s_tot


def accumulate_coral_cards(
    existing: dict[tuple[str, str], CoralScoredCard],
    new_cards: list[CoralScoredCard],
) -> dict[tuple[str, str], CoralScoredCard]:
    out = dict(existing)
    for item in new_cards:
        if not item.kept:
            continue
        key = item.dedupe_key
        prev = out.get(key)
        if prev is None or item.s_tot > prev.s_tot:
            out[key] = item
    return out


def coral_question_text(row: dict[str, Any], *, translated_query: dict[str, Any] | None = None) -> str:
    original = (
        f"Subject: {row.get('subject', '')}\n"
        f"Language: {row.get('language') or row.get('_language') or ''}\n"
        f"Question:\n{row.get('question', '')}\n\n"
        f"Options:\n{format_options(row.get('options') or {})}"
    )
    if not translated_query:
        return original
    translated_options = translated_query.get("options") if isinstance(translated_query, dict) else {}
    if not isinstance(translated_options, dict):
        translated_options = {}
    labels = option_labels(row.get("options") or {})
    translated_option_text = "\n".join(
        f"{label}. {translated_options.get(label) or (row.get('options') or {}).get(label) or ''}"
        for label in labels
    )
    translated = (
        f"Subject: {row.get('subject', '')}\n"
        "Language: English translation for card matching\n"
        f"Question:\n{translated_query.get('question', '')}\n\n"
        f"Options:\n{translated_option_text}"
    )
    return (
        "Primary English problem statement for matching English usage cards:\n"
        f"{translated}\n\n"
        "Original low-resource problem statement for label consistency:\n"
        f"{original}"
    )


def coral_retrieval_query(row: dict[str, Any]) -> str:
    return f"{row.get('question', '')}\n\nOptions:\n{format_options(row.get('options') or {})}".strip()


def estimate_token_count(text: str) -> int:
    # Stable local approximation used for trace accounting without requiring
    # the generation tokenizer inside the retrieval/planning code path.
    text = text or ""
    if not text:
        return 0
    return max(1, len(text) // 4)


def coral_planner_messages(
    row: dict[str, Any],
    *,
    language_pool: list[str],
    max_corpora: int,
    previous_query: str | None = None,
    previous_languages: list[str] | None = None,
    sufficiency_reason: str | None = None,
) -> list[dict[str, str]]:
    language = supported_language(row)
    is_replan = bool(previous_query or previous_languages or sufficiency_reason)
    schema = (
        '{"language_names":["hi","en"],"rewritten_query":"..."}'
        if is_replan
        else '{"language_names":["hi","en"]}'
    )
    extra = ""
    if is_replan:
        extra = (
            f"\nPrevious query:\n{previous_query or ''}\n"
            f"Previous language_names: {json.dumps(previous_languages or [], ensure_ascii=False)}\n"
            f"Sufficiency critic reason:\n{sufficiency_reason or ''}\n"
            "Because evidence was insufficient, you must change either language_names or rewritten_query."
        )
    return [
        {
            "role": "system",
            "content": (
                "You are the CORAL-Wikipag planner. Select retrieval corpora for an inference-only "
                "multiple-choice baseline. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Language pool: {json.dumps(language_pool)}\n"
                f"Question language: {language}\n"
                f"Maximum corpora: {max_corpora}\n"
                "Rules: include the question language; choose at most the maximum corpora; "
                "use only languages from the pool.\n\n"
                f"{coral_question_text(row)}"
                f"{extra}\n\n"
                f"Return exactly this JSON schema: {schema}"
            ),
        },
    ]


def coral_critic_messages(
    row: dict[str, Any],
    *,
    card: CoralCard,
    query: str,
    translated_query: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the CORAL-Wikipag card critic. Score exactly one usage card for one "
                "multiple-choice question. Do not answer the question. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Retrieval query:\n{query}\n\n"
                f"{coral_question_text(row, translated_query=translated_query)}\n\n"
                "Usage card to score:\n"
                f"Corpus language: {card.language}\n"
                f"Document ID: {card.document_id}\n"
                f"Concept ID: {card.concept_id}\n"
                f"Similarity: {card.score:.6f}\n"
                f"{card.text[:3000]}\n\n"
                "Score each dimension from 0 to 5:\n"
                "- relevance: does this card address the question/options?\n"
                "- usefulness: would it help choose the answer?\n"
                "- clarity_specificity: is it specific enough to apply?\n"
                "- compatibility: is it compatible with the question context and options?\n\n"
                'Return exactly JSON: {"scores":{"relevance":0,"usefulness":0,'
                '"clarity_specificity":0,"compatibility":0},"critique":"..."}'
            ),
        },
    ]


def coral_batch_critic_messages(
    row: dict[str, Any],
    *,
    cards: list[CoralCard],
    query: str,
    translated_query: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    compact_cards = [
        {
            "id": idx,
            "corpus_language": card.language,
            "document_id": card.document_id,
            "concept_id": card.concept_id,
            "usage_id": card.usage_id,
            "similarity": round(float(card.score), 6),
            "text": card.text[:2500],
        }
        for idx, card in enumerate(cards)
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are the CORAL-Wikipag batch card critic. Score each usage card for one "
                "multiple-choice question. Do not answer the question. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Retrieval query:\n{query}\n\n"
                f"{coral_question_text(row, translated_query=translated_query)}\n\n"
                "Score every card independently from 0 to 5:\n"
                "- relevance: does this card address the question/options?\n"
                "- usefulness: would it help choose the answer?\n"
                "- clarity_specificity: is it specific enough to apply?\n"
                "- compatibility: is it compatible with the question context and options?\n\n"
                "Return exactly JSON with one item per input id: "
                '{"cards":[{"id":0,"scores":{"relevance":0,"usefulness":0,'
                '"clarity_specificity":0,"compatibility":0},"critique":"..."}]}\n\n'
                f"Cards:\n{json.dumps(compact_cards, ensure_ascii=False)}"
            ),
        },
    ]


def coral_sufficiency_messages(
    row: dict[str, Any],
    *,
    kept_cards: list[CoralScoredCard],
    translated_query: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    evidence = "\n\n".join(format_coral_card_for_prompt(item, include_score=True) for item in kept_cards[:8])
    if not evidence:
        evidence = "No kept usage cards."
    return [
        {
            "role": "system",
            "content": (
                "You are the CORAL-Wikipag sufficiency critic. Judge only whether the kept evidence "
                "is enough for answering. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{coral_question_text(row, translated_query=translated_query)}\n\n"
                f"Kept evidence:\n{evidence}\n\n"
                'Return exactly JSON: {"enough_documents":true,"reason":"..."}'
            ),
        },
    ]


def coral_solver_messages(
    row: dict[str, Any],
    *,
    cards: list[CoralScoredCard],
    triples: list[dict[str, Any]],
    fallback_reason: str | None = None,
    translated_query: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    card_block = "\n\n".join(format_coral_card_for_prompt(item, include_score=True) for item in cards)
    if not card_block:
        card_block = "No verified CORAL-Wikipag usage cards."
    triple_block = "\n".join(
        f"- ({item.get('source_concept_id')}, {item.get('relation')}, {item.get('target_concept_id')})"
        for item in triples
    )
    if not triple_block:
        triple_block = "No related concept triples."
    labels = ", ".join(option_labels(row.get("options") or {}))
    return [
        {
            "role": "system",
            "content": (
                "You answer multilingual multiple-choice questions using CORAL-Wikipag usage cards "
                "and concept triples when available. Hidden reasoning is disabled. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Method: CORAL-Wikipag\n"
                f"{coral_question_text(row, translated_query=translated_query)}\n\n"
                f"Fallback reason: {fallback_reason or 'none'}\n\n"
                f"Final usage cards:\n{card_block}\n\n"
                f"Concept triples:\n{triple_block}\n\n"
                f"Choose the single best answer from {labels}. "
                'Return only JSON: {"answer":"A"}'
            ),
        },
    ]


def coral_direct_solver_messages(
    row: dict[str, Any],
    *,
    translated_query: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    labels = ", ".join(option_labels(row.get("options") or {}))
    return [
        {
            "role": "system",
            "content": (
                "You answer multilingual multiple-choice exam questions without retrieved evidence. "
                "Hidden reasoning is disabled. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{coral_question_text(row, translated_query=translated_query)}\n\n"
                f"Choose the single best answer from {labels}. "
                'Return only JSON: {"answer":"A"}'
            ),
        },
    ]


def coral_answer_gate_messages(
    row: dict[str, Any],
    *,
    cards: list[CoralScoredCard],
    translated_query: dict[str, Any] | None,
    direct_answer: str | None,
    coral_answer: str | None,
) -> list[dict[str, str]]:
    card_block = "\n\n".join(format_coral_card_for_prompt(item, include_score=True) for item in cards) or "No cards."
    labels = ", ".join(option_labels(row.get("options") or {}))
    return [
        {
            "role": "system",
            "content": (
                "You are a conservative gate for CORAL-Wikipag usage cards. "
                "Choose between the direct answer and the card-based answer only. "
                "Hidden reasoning is disabled. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{coral_question_text(row, translated_query=translated_query)}\n\n"
                f"Direct no-card answer: {direct_answer or 'INVALID'}\n"
                f"CORAL card-based answer: {coral_answer or 'INVALID'}\n\n"
                f"Candidate CORAL cards:\n{card_block}\n\n"
                "Select the CORAL card-based answer only if at least one card gives a directly applicable rule, "
                "trigger, pitfall, or concept boundary that supports that answer or eliminates the direct answer. "
                "Merely mentioning the same broad concept is not enough. For fill-in-the-blank questions, "
                "statement truth-value questions, anatomy landmark questions, numeric/list-option questions, "
                "or questions where options differ by small qualifiers, the card must contain the decisive fact "
                "or a direct paraphrase connecting the question to the chosen option. If cards are generic, "
                "weakly related, incomplete, conflicting, or only provide a broad concept boundary, select the "
                "direct no-card answer. "
                f"The final answer must be one of {labels}. "
                'Return exactly JSON: {"answer":"A","selected_source":"direct|coral",'
                '"evidence_status":"direct_card_support|weak_or_irrelevant|conflicting|invalid_candidate"}'
            ),
        },
    ]


def format_coral_card_for_prompt(item: CoralScoredCard, *, include_score: bool) -> str:
    card = item.card
    prefix = (
        f"[{card.language}:{card.document_id}] subject={card.subject} "
        f"concept_id={card.concept_id} usage_id={card.usage_id}"
    )
    if include_score:
        prefix += f" s_tot={item.s_tot:.3f} similarity={card.score:.6f}"
    return f"{prefix}\n{card.text[:3500]}"


def normalize_coral_languages(raw: Any, *, language_pool: list[str], question_language: str, max_corpora: int) -> list[str]:
    if not isinstance(raw, list):
        raw = []
    seen: set[str] = set()
    out: list[str] = []
    for item in [question_language, *raw]:
        language = str(item).strip()
        if language in language_pool and language not in seen:
            seen.add(language)
            out.append(language)
        if len(out) >= max_corpora:
            break
    if not out:
        out = [question_language if question_language in language_pool else language_pool[0]]
    return out[:max_corpora]


def validate_coral_scores(payload: dict[str, Any]) -> dict[str, Any]:
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("missing scores object")
    normalized: dict[str, float] = {}
    for key in ["relevance", "usefulness", "clarity_specificity", "compatibility"]:
        try:
            value = float(scores.get(key))
        except Exception as exc:
            raise ValueError(f"invalid score {key}") from exc
        if not 0.0 <= value <= 5.0:
            raise ValueError(f"score {key} out of range: {value}")
        normalized[key] = value
    critique = str(payload.get("critique") or "").strip()
    return {"scores": normalized, "critique": critique}


def validate_coral_batch_scores(payload: dict[str, Any]) -> dict[str, Any]:
    raw_cards = payload.get("cards")
    if not isinstance(raw_cards, list):
        raise ValueError("missing cards list")
    out: dict[int, dict[str, Any]] = {}
    for item in raw_cards:
        if not isinstance(item, dict):
            continue
        try:
            card_id = int(item.get("id"))
        except Exception:
            continue
        out[card_id] = validate_coral_scores(item)
    if not out:
        raise ValueError("empty card score list")
    return {"cards": out}


def validate_translated_query(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question") or "").strip()
    options = payload.get("options")
    if not question:
        raise ValueError("missing translated question")
    if not isinstance(options, dict):
        raise ValueError("missing translated options")
    return {"question": question, "options": options}


class CoralUsageBank:
    def __init__(self, *, bank_dir: Path, model_dir: Path, device: str, cache: Any | None = None) -> None:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if not bank_dir.exists():
            raise FileNotFoundError(bank_dir)
        self.bank_dir = bank_dir
        self.cache = cache
        self.faiss = faiss
        self.np = np
        self.model = SentenceTransformer(str(model_dir), device=device)
        self._subject_indexes: dict[str, dict[str, Any]] = {}
        self._cards_by_usage_id: dict[str, dict[str, Any]] | None = None
        self._relations_by_concept_id: dict[str, list[dict[str, Any]]] | None = None

    def close(self) -> None:
        return None

    def _load_cards(self) -> dict[str, dict[str, Any]]:
        if self._cards_by_usage_id is None:
            cards: dict[str, dict[str, Any]] = {}
            path = self.bank_dir / "usage_cards.jsonl"
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    usage_id = str(row.get("usage_id") or "")
                    if usage_id:
                        cards[usage_id] = row
            self._cards_by_usage_id = cards
        return self._cards_by_usage_id

    def _load_relations(self) -> dict[str, list[dict[str, Any]]]:
        if self._relations_by_concept_id is None:
            relations: dict[str, list[dict[str, Any]]] = {}
            path = self.bank_dir / "concept_relations.jsonl"
            if path.exists():
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        for key in ["source_concept_id", "target_concept_id"]:
                            concept_id = str(row.get(key) or "")
                            if concept_id:
                                relations.setdefault(concept_id, []).append(row)
            self._relations_by_concept_id = relations
        return self._relations_by_concept_id

    def _load_subject(self, subject: str) -> dict[str, Any]:
        if subject in self._subject_indexes:
            return self._subject_indexes[subject]
        subject_dir = self.bank_dir / "usage_indexes" / subject
        index_path = subject_dir / "usage_index.faiss"
        ids_path = subject_dir / "usage_index_ids.jsonl"
        rows_path = subject_dir / "usage_index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        index = self.faiss.read_index(str(index_path))
        ids = [json.loads(line) for line in ids_path.open(encoding="utf-8") if line.strip()]
        rows = [json.loads(line) for line in rows_path.open(encoding="utf-8") if line.strip()]
        payload = {"index": index, "ids": ids, "rows": rows, "dim": int(index.d)}
        self._subject_indexes[subject] = payload
        return payload

    def _encode(self, query: str, dim: int) -> Any:
        kwargs: dict[str, Any] = {
            "prompt_name": "query",
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        try:
            vector = self.model.encode([query], truncate_dim=dim, **kwargs)
        except TypeError:
            vector = self.model.encode([query], **kwargs)
            if vector.shape[1] > dim:
                vector = vector[:, :dim]
                vector = vector / self.np.maximum(self.np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
        vector = self.np.asarray(vector, dtype=self.np.float32)
        if vector.shape != (1, dim):
            raise ValueError(f"embedding shape {vector.shape} != (1, {dim})")
        return vector

    def retrieve(self, *, subject: str, query: str, language: str, top_k: int) -> list[CoralCard]:
        payload = {"subject": subject, "query": query, "language": language, "top_k": int(top_k)}
        if self.cache is not None:
            cached = self.cache.get("coral_wikipag_retrieval", payload)
            if cached is not None:
                return [coral_card_from_payload(item) for item in cached.get("results", [])]
        loaded = self._load_subject(subject)
        query_vec = self._encode(query, int(loaded["dim"]))
        scores, hits = loaded["index"].search(query_vec, int(top_k))
        cards_by_usage = self._load_cards()
        results: list[CoralCard] = []
        for rank, (score, hit) in enumerate(zip(scores[0].tolist(), hits[0].tolist()), start=1):
            if hit < 0:
                continue
            id_row = loaded["ids"][int(hit)]
            row_id = int(id_row.get("row_id", hit))
            index_row = loaded["rows"][row_id]
            usage_id = str(index_row.get("usage_id") or id_row.get("usage_id") or "")
            concept_id = str(index_row.get("concept_id") or id_row.get("concept_id") or "")
            detail = cards_by_usage.get(usage_id, {})
            text = build_card_text(index_row, detail)
            results.append(
                CoralCard(
                    language=language,
                    document_id=usage_id or f"{subject}:{row_id}",
                    subject=subject,
                    concept_id=concept_id,
                    usage_id=usage_id,
                    index_key=str(index_row.get("index_key") or ""),
                    payload=str(index_row.get("payload") or ""),
                    text=text,
                    score=float(score),
                    rank=rank,
                    metadata={"row_id": row_id, "status": index_row.get("status")},
                )
            )
        if self.cache is not None:
            self.cache.set("coral_wikipag_retrieval", payload, {"results": [coral_card_to_payload(item) for item in results]})
        return results

    def triples_for_concepts(self, concept_ids: list[str], *, max_triples: int) -> list[dict[str, Any]]:
        relations = self._load_relations()
        seen: set[str] = set()
        triples: list[dict[str, Any]] = []
        for concept_id in concept_ids:
            for row in relations.get(concept_id, []):
                key = json.dumps(
                    {
                        "s": row.get("source_concept_id"),
                        "r": row.get("relation"),
                        "t": row.get("target_concept_id"),
                    },
                    sort_keys=True,
                )
                if key in seen:
                    continue
                seen.add(key)
                triples.append(row)
                if len(triples) >= max_triples:
                    return triples
        return triples


class MockCoralUsageBank:
    def __init__(self) -> None:
        self.calls = 0

    def close(self) -> None:
        return None

    def retrieve(self, *, subject: str, query: str, language: str, top_k: int) -> list[CoralCard]:
        self.calls += 1
        return [
            CoralCard(
                language=language,
                document_id=f"mock_doc_{idx}",
                subject=subject,
                concept_id=f"MOCK-CONCEPT-{idx}",
                usage_id=f"mock_usage_{idx}",
                index_key=f"Subject: {subject}\nConcept: mock concept {idx}",
                payload="Decision Procedure:\n- Use the mock card.",
                text=f"Subject: {subject}\nConcept: mock concept {idx}\nDecision Procedure:\n- Use the mock card.",
                score=1.0 - idx * 0.01,
                rank=idx + 1,
                metadata={"mock": True},
            )
            for idx in range(max(0, int(top_k)))
        ]

    def triples_for_concepts(self, concept_ids: list[str], *, max_triples: int) -> list[dict[str, Any]]:
        triples = [
            {
                "source_concept_id": concept_id,
                "relation": "RELATED_TO",
                "target_concept_id": "MOCK-TARGET",
                "confidence": 1.0,
            }
            for concept_id in concept_ids
        ]
        return triples[:max_triples]


def build_card_text(index_row: dict[str, Any], detail: dict[str, Any]) -> str:
    if not detail:
        return f"{index_row.get('index_key', '')}\n\n{index_row.get('payload', '')}".strip()
    lines = [
        f"Subject: {detail.get('subject', index_row.get('subject', ''))}",
        f"Concept: {detail.get('concept', '')}",
        f"Usage Pattern: {detail.get('usage_pattern', '')}",
        f"Usage Signature: {detail.get('usage_signature', '')}",
        f"Concept Boundary: {detail.get('concept_boundary', '')}",
    ]
    for label, key in [
        ("Trigger Conditions", "trigger_conditions"),
        ("Decision Procedure", "decision_procedure"),
        ("Failure Boundaries", "failure_boundaries"),
        ("Verification Rules", "verification_rules"),
    ]:
        value = detail.get(key)
        if isinstance(value, list):
            lines.append(label + ":\n" + "\n".join(f"- {item}" for item in value))
        elif value:
            lines.append(f"{label}: {value}")
    return "\n".join(line for line in lines if line.strip())


def coral_card_to_payload(card: CoralCard) -> dict[str, Any]:
    return {
        "language": card.language,
        "document_id": card.document_id,
        "subject": card.subject,
        "concept_id": card.concept_id,
        "usage_id": card.usage_id,
        "index_key": card.index_key,
        "payload": card.payload,
        "text": card.text,
        "score": card.score,
        "rank": card.rank,
        "metadata": card.metadata,
    }


def coral_card_from_payload(item: dict[str, Any]) -> CoralCard:
    return CoralCard(
        language=str(item.get("language") or ""),
        document_id=str(item.get("document_id") or ""),
        subject=str(item.get("subject") or ""),
        concept_id=str(item.get("concept_id") or ""),
        usage_id=str(item.get("usage_id") or ""),
        index_key=str(item.get("index_key") or ""),
        payload=str(item.get("payload") or ""),
        text=str(item.get("text") or ""),
        score=float(item.get("score") or 0.0),
        rank=int(item.get("rank") or 0),
        metadata=dict(item.get("metadata") or {}),
    )


async def model_json_with_retry(
    model: Any,
    messages_factory: Callable[[], list[dict[str, str]]],
    *,
    namespace: str,
    max_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
    retries: int = 1,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            payload = await model.json(
                messages_factory(),
                namespace=f"{namespace}.try{attempt}",
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return validator(payload)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"model_json_with_retry failed: {type(last_error).__name__}: {last_error}")


def validate_sufficiency(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload.get("enough_documents"), bool):
        raise ValueError("missing enough_documents bool")
    return {
        "enough_documents": bool(payload.get("enough_documents")),
        "reason": str(payload.get("reason") or "").strip(),
    }


async def run_coral_wikipag(
    row: dict[str, Any],
    *,
    model: Any,
    bank: CoralUsageBank,
    config: dict[str, Any],
    row_key: str,
    parse_answer_label: Callable[[dict[str, Any], dict[str, Any]], tuple[str | None, bool]],
) -> dict[str, Any]:
    started = time.perf_counter()
    question_language = supported_language(row)
    language_pool = list(config.get("coral_language_pool") or DEFAULT_LANGUAGE_POOL)
    if question_language not in language_pool:
        language_pool = [question_language, *language_pool]
    max_corpora = int(config.get("coral_max_corpora") or 3)
    top_k_per_corpus = int(config.get("coral_top_k_per_corpus") or 5)
    final_top_k = int(config.get("coral_final_top_k") or config.get("final_top_k") or 5)
    max_rounds = int(config.get("coral_max_rounds") or 3)
    planner_temperature = float(config.get("coral_planner_temperature") if config.get("coral_planner_temperature") is not None else 0.6)
    critic_temperature = float(config.get("coral_critic_temperature") if config.get("coral_critic_temperature") is not None else 0.6)
    generator_temperature = float(config.get("coral_generator_temperature") if config.get("coral_generator_temperature") is not None else 0.0)
    min_each = float(config.get("coral_critic_min_score") or 2.0)
    min_total = float(config.get("coral_min_total_score") or 6.0)
    max_card_chars = int(config.get("coral_max_card_chars") or 3500)
    max_triples = int(config.get("coral_max_triples") or 20)
    use_batch_critic = bool(config.get("coral_batch_critic"))
    translate_query = bool(config.get("coral_translate_query"))
    dual_query_retrieval = bool(config.get("coral_dual_query_retrieval"))
    subject = str(row.get("subject") or "")
    translated_query_payload: dict[str, Any] | None = None
    original_query = coral_retrieval_query(row)
    current_query = original_query
    if translate_query and question_language != "en":
        try:
            translated_query_payload = await model_json_with_retry(
                model,
                lambda: translate_query_messages(row),
                namespace=f"{row_key}.coral.translate_query",
                max_tokens=int(config.get("coral_translate_query_max_tokens") or 512),
                temperature=float(config.get("coral_translate_query_temperature") or 0.0),
                validator=validate_translated_query,
                retries=1,
            )
            current_query = english_query_from_translated_payload(translated_query_payload)
        except Exception:
            translated_query_payload = None
            current_query = original_query
    previous_languages: list[str] | None = None
    sufficiency_reason: str | None = None
    accumulated: dict[tuple[str, str], CoralScoredCard] = {}
    trace_rounds: list[dict[str, Any]] = []
    stop_reason = "max_rounds"

    for round_idx in range(1, max_rounds + 1):
        is_replan = round_idx > 1

        def planner_factory() -> list[dict[str, str]]:
            return coral_planner_messages(
                row,
                language_pool=language_pool,
                max_corpora=max_corpora,
                previous_query=current_query if is_replan else None,
                previous_languages=previous_languages if is_replan else None,
                sufficiency_reason=sufficiency_reason if is_replan else None,
            )

        def planner_validator(payload: dict[str, Any]) -> dict[str, Any]:
            languages = normalize_coral_languages(
                payload.get("language_names"),
                language_pool=language_pool,
                question_language=question_language,
                max_corpora=max_corpora,
            )
            rewritten = str(payload.get("rewritten_query") or "").strip()
            if is_replan:
                same_langs = languages == (previous_languages or [])
                same_query = not rewritten or rewritten == current_query
                if same_langs and same_query:
                    raise ValueError("planner did not modify languages or query")
            return {"language_names": languages, "rewritten_query": rewritten}

        try:
            plan = await model_json_with_retry(
                model,
                planner_factory,
                namespace=f"{row_key}.coral.round{round_idx}.planner",
                max_tokens=int(config.get("coral_planner_max_tokens") or 256),
                temperature=planner_temperature,
                validator=planner_validator,
                retries=1,
            )
        except Exception as exc:
            fallback_query = f"{current_query}\n\nNeed more evidence: {sufficiency_reason or type(exc).__name__}"
            plan = {
                "language_names": normalize_coral_languages(
                    previous_languages or [question_language],
                    language_pool=language_pool,
                    question_language=question_language,
                    max_corpora=max_corpora,
                ),
                "rewritten_query": fallback_query if is_replan else "",
                "planner_error": f"{type(exc).__name__}: {exc}",
            }
        languages = list(plan["language_names"])
        if is_replan and plan.get("rewritten_query"):
            current_query = str(plan["rewritten_query"])
        previous_languages = languages

        retrievals: list[dict[str, Any]] = []
        candidates: list[CoralCard] = []
        for language in languages:
            query_specs = [("primary", current_query)]
            if dual_query_retrieval and original_query != current_query:
                query_specs.append(("original", original_query))
            cards_by_key: dict[tuple[str, str], CoralCard] = {}
            raw_results: list[dict[str, Any]] = []
            for query_source, query_text in query_specs:
                cards = bank.retrieve(subject=subject, query=query_text, language=language, top_k=top_k_per_corpus)
                for card in cards:
                    key = (card.language, card.document_id)
                    prev = cards_by_key.get(key)
                    if prev is None or card.score > prev.score:
                        metadata = dict(card.metadata)
                        metadata["query_source"] = query_source
                        cards_by_key[key] = CoralCard(
                            language=card.language,
                            document_id=card.document_id,
                            subject=card.subject,
                            concept_id=card.concept_id,
                            usage_id=card.usage_id,
                            index_key=card.index_key,
                            payload=card.payload,
                            text=card.text,
                            score=card.score,
                            rank=card.rank,
                            metadata=metadata,
                        )
                raw_results.append(
                    {
                        "query_source": query_source,
                        "query_preview": query_text[:300],
                        "results": [
                            {
                                "document_id": card.document_id,
                                "usage_id": card.usage_id,
                                "concept_id": card.concept_id,
                                "score": card.score,
                                "rank": card.rank,
                                "text_preview": card.text[:300],
                            }
                            for card in cards
                        ],
                    }
                )
            combined_limit = top_k_per_corpus * len(query_specs) if len(query_specs) > 1 else top_k_per_corpus
            cards = sorted(cards_by_key.values(), key=lambda item: item.score, reverse=True)[:combined_limit]
            candidates.extend(cards)
            retrievals.append(
                {
                    "language": language,
                    "top_k": top_k_per_corpus,
                    "queries": raw_results,
                    "results": [
                        {
                            "document_id": card.document_id,
                            "usage_id": card.usage_id,
                            "concept_id": card.concept_id,
                            "score": card.score,
                            "rank": card.rank,
                            "query_source": card.metadata.get("query_source"),
                            "text_preview": card.text[:300],
                        }
                        for card in cards
                    ],
                }
            )

        limited_cards = [
            CoralCard(
                language=card.language,
                document_id=card.document_id,
                subject=card.subject,
                concept_id=card.concept_id,
                usage_id=card.usage_id,
                index_key=card.index_key,
                payload=card.payload,
                text=card.text[:max_card_chars],
                score=card.score,
                rank=card.rank,
                metadata=card.metadata,
            )
            for card in candidates
        ]

        scored: list[CoralScoredCard] = []
        if use_batch_critic and limited_cards:
            try:
                batch_payload = await model_json_with_retry(
                    model,
                    lambda: coral_batch_critic_messages(
                        row,
                        cards=limited_cards,
                        query=current_query,
                        translated_query=translated_query_payload,
                    ),
                    namespace=f"{row_key}.coral.round{round_idx}.critic_batch",
                    max_tokens=int(config.get("coral_critic_max_tokens") or 2048),
                    temperature=critic_temperature,
                    validator=validate_coral_batch_scores,
                    retries=1,
                )
                scored_by_id = dict(batch_payload["cards"])
            except Exception as exc:
                scored_by_id = {
                    idx: {
                        "scores": {
                            "relevance": 0.0,
                            "usefulness": 0.0,
                            "clarity_specificity": 0.0,
                            "compatibility": 0.0,
                        },
                        "critique": f"critic_batch_error:{type(exc).__name__}: {exc}",
                    }
                    for idx in range(len(limited_cards))
                }
            for idx, limited_card in enumerate(limited_cards):
                critique_payload = scored_by_id.get(
                    idx,
                    {
                        "scores": {
                            "relevance": 0.0,
                            "usefulness": 0.0,
                            "clarity_specificity": 0.0,
                            "compatibility": 0.0,
                        },
                        "critique": "critic_batch_missing_score",
                    },
                )
                kept, keep_reason, s_tot = coral_card_passes(
                    critique_payload["scores"],
                    min_each=min_each,
                    min_total=min_total,
                )
                scored.append(
                    CoralScoredCard(
                        card=limited_card,
                        scores=critique_payload["scores"],
                        critique=critique_payload.get("critique", ""),
                        s_tot=s_tot,
                        kept=kept,
                        keep_reason=keep_reason,
                        round_idx=round_idx,
                    )
                )
        else:
            for limited_card in limited_cards:

                async def score_one() -> dict[str, Any]:
                    return await model_json_with_retry(
                        model,
                        lambda limited_card=limited_card: coral_critic_messages(
                            row,
                            card=limited_card,
                            query=current_query,
                            translated_query=translated_query_payload,
                        ),
                        namespace=f"{row_key}.coral.round{round_idx}.critic.{limited_card.language}.{limited_card.document_id}",
                        max_tokens=int(config.get("coral_critic_max_tokens") or 256),
                        temperature=critic_temperature,
                        validator=validate_coral_scores,
                        retries=1,
                    )

                try:
                    critique_payload = await score_one()
                    kept, keep_reason, s_tot = coral_card_passes(
                        critique_payload["scores"],
                        min_each=min_each,
                        min_total=min_total,
                    )
                    scored.append(
                        CoralScoredCard(
                            card=limited_card,
                            scores=critique_payload["scores"],
                            critique=critique_payload.get("critique", ""),
                            s_tot=s_tot,
                            kept=kept,
                            keep_reason=keep_reason,
                            round_idx=round_idx,
                        )
                    )
                except Exception as exc:
                    scored.append(
                        CoralScoredCard(
                            card=limited_card,
                            scores={"relevance": 0.0, "usefulness": 0.0, "clarity_specificity": 0.0, "compatibility": 0.0},
                            critique=f"critic_error:{type(exc).__name__}: {exc}",
                            s_tot=0.0,
                            kept=False,
                            keep_reason="critic_error",
                            round_idx=round_idx,
                        )
                    )
        accumulated = accumulate_coral_cards(accumulated, scored)
        kept_so_far = sorted(accumulated.values(), key=lambda item: item.s_tot, reverse=True)

        try:
            sufficiency = await model_json_with_retry(
                model,
                lambda: coral_sufficiency_messages(row, kept_cards=kept_so_far, translated_query=translated_query_payload),
                namespace=f"{row_key}.coral.round{round_idx}.sufficiency",
                max_tokens=int(config.get("coral_sufficiency_max_tokens") or 256),
                temperature=critic_temperature,
                validator=validate_sufficiency,
                retries=1,
            )
        except Exception as exc:
            sufficiency = {
                "enough_documents": False,
                "reason": f"sufficiency_error:{type(exc).__name__}: {exc}",
            }
        sufficiency_reason = str(sufficiency.get("reason") or "")
        trace_rounds.append(
            {
                "round": round_idx,
                "query": current_query,
                "token_counts": {
                    "query_estimated_tokens": estimate_token_count(current_query),
                    "retrieved_card_estimated_tokens": sum(estimate_token_count(card.text) for card in candidates),
                    "kept_card_estimated_tokens": sum(estimate_token_count(item.card.text) for item in kept_so_far),
                },
                "planner": plan,
                "language_names": languages,
                "retrieval": retrievals,
                "scored_cards": [scored_card_to_trace(item) for item in scored],
                "accumulated_kept_count": len(accumulated),
                "sufficiency": sufficiency,
            }
        )
        if sufficiency.get("enough_documents"):
            stop_reason = "sufficient"
            break

    final_cards = sorted(accumulated.values(), key=lambda item: item.s_tot, reverse=True)[:final_top_k]
    fallback_reason = None
    if not final_cards:
        fallback_reason = "no_qualified_coral_wikipag_cards"
    triples = bank.triples_for_concepts([item.card.concept_id for item in final_cards], max_triples=max_triples)
    def answer_validator(payload: dict[str, Any]) -> dict[str, Any]:
        pred, valid = parse_answer_label(payload, row)
        if not valid or pred is None:
            raise ValueError("invalid final answer JSON")
        return payload

    answer_payload = await model_json_with_retry(
        model,
        lambda: coral_solver_messages(
            row,
            cards=final_cards,
            triples=triples,
            fallback_reason=fallback_reason,
            translated_query=translated_query_payload,
        ),
        namespace=f"{row_key}.coral.answer",
        max_tokens=int(config.get("answer_max_tokens") or 512),
        temperature=generator_temperature,
        top_p=float(config.get("coral_generator_top_p") or 1.0),
        validator=answer_validator,
        retries=1,
    )
    coral_prediction, coral_valid = parse_answer_label(answer_payload, row)
    pred, valid = coral_prediction, coral_valid
    direct_payload: dict[str, Any] | None = None
    direct_prediction: str | None = None
    direct_valid = False
    gate_payload: dict[str, Any] | None = None
    gate_selected_source: str | None = None
    gate_evidence_status: str | None = None
    if bool(config.get("coral_adaptive_direct_gate")):
        try:
            direct_payload = await model_json_with_retry(
                model,
                lambda: coral_direct_solver_messages(row, translated_query=translated_query_payload),
                namespace=f"{row_key}.coral.direct_answer",
                max_tokens=int(config.get("answer_max_tokens") or 512),
                temperature=generator_temperature,
                top_p=float(config.get("coral_generator_top_p") or 1.0),
                validator=answer_validator,
                retries=1,
            )
            direct_prediction, direct_valid = parse_answer_label(direct_payload, row)
        except Exception as exc:
            direct_payload = {"error": f"{type(exc).__name__}: {exc}"}
            direct_prediction, direct_valid = None, False
        try:
            gate_payload = await model_json_with_retry(
                model,
                lambda: coral_answer_gate_messages(
                    row,
                    cards=final_cards,
                    translated_query=translated_query_payload,
                    direct_answer=direct_prediction,
                    coral_answer=coral_prediction,
                ),
                namespace=f"{row_key}.coral.answer_gate",
                max_tokens=int(config.get("gate_max_tokens") or 256),
                temperature=critic_temperature,
                validator=answer_validator,
                retries=1,
            )
            gated_pred, gated_valid = parse_answer_label(gate_payload, row)
            selected_raw = str(gate_payload.get("selected_source") or "").strip().lower()
            gate_selected_source = selected_raw if selected_raw in {"direct", "coral"} else None
            gate_evidence_status = str(gate_payload.get("evidence_status") or "").strip()
        except Exception as exc:
            gate_payload = {"error": f"{type(exc).__name__}: {exc}"}
            gated_pred, gated_valid = None, False
        if gated_valid and gated_pred in {direct_prediction, coral_prediction}:
            pred, valid = gated_pred, True
        elif gate_selected_source == "coral" and coral_valid:
            pred, valid = coral_prediction, True
        elif direct_valid:
            pred, valid = direct_prediction, True
            gate_selected_source = gate_selected_source or "direct"
        else:
            pred, valid = coral_prediction, coral_valid
            gate_selected_source = gate_selected_source or "coral"
    gold = str(row.get("answer") or "").strip().upper()
    return {
        "eval_key": row_key,
        "method": "coral_wikipag",
        "method_name": "CORAL-Wikipag",
        "dataset": row.get("_dataset") or row.get("dataset") or "global_mmlu",
        "language": row.get("language") or row.get("_language"),
        "subject": row.get("subject"),
        "sample_id": row.get("sample_id") or row.get("question_id") or row.get("id"),
        "answer": gold,
        "prediction": pred,
        "valid": valid,
        "correct": bool(valid and pred == gold),
        "coral_prediction": coral_prediction,
        "coral_valid": coral_valid,
        "direct_prediction": direct_prediction,
        "direct_valid": direct_valid,
        "direct_payload": direct_payload,
        "gate_payload": gate_payload,
        "gate_selected_source": gate_selected_source,
        "gate_evidence_status": gate_evidence_status,
        "retrieval_query": coral_retrieval_query(row),
        "final_query": current_query,
        "translated_query": translated_query_payload,
        "stop_reason": stop_reason,
        "fallback_reason": fallback_reason,
        "final_top_k": final_top_k,
        "coral_config": {
            "retrieval_source": config.get("coral_retrieval_source") or "cards",
            "language_pool": language_pool,
            "max_corpora": max_corpora,
            "top_k_per_corpus": top_k_per_corpus,
            "max_rounds": max_rounds,
            "min_each": min_each,
            "min_total": min_total,
            "batch_critic": use_batch_critic,
            "translate_query": translate_query,
            "dual_query_retrieval": dual_query_retrieval,
            "adaptive_direct_gate": bool(config.get("coral_adaptive_direct_gate")),
        },
        "final_cards": [scored_card_to_trace(item) for item in final_cards],
        "token_counts": {
            "retrieval_query_estimated_tokens": estimate_token_count(coral_retrieval_query(row)),
            "final_query_estimated_tokens": estimate_token_count(current_query),
            "final_card_estimated_tokens": sum(estimate_token_count(item.card.text) for item in final_cards),
            "triple_count": len(triples),
        },
        "triples": triples,
        "answer_payload": answer_payload,
        "trace": {"rounds": trace_rounds},
        "latency_s": time.perf_counter() - started,
    }


def scored_card_to_trace(item: CoralScoredCard) -> dict[str, Any]:
    return {
        "round": item.round_idx,
        "language": item.card.language,
        "document_id": item.card.document_id,
        "usage_id": item.card.usage_id,
        "concept_id": item.card.concept_id,
        "subject": item.card.subject,
        "similarity": item.card.score,
        "rank": item.card.rank,
        "scores": item.scores,
        "s_tot": item.s_tot,
        "kept": item.kept,
        "keep_reason": item.keep_reason,
        "critique": item.critique,
        "text_preview": item.card.text[:500],
        "token_counts": {
            "card_text_estimated_tokens": estimate_token_count(item.card.text),
            "critique_estimated_tokens": estimate_token_count(item.critique),
        },
    }
