from __future__ import annotations

import csv
import json
import re
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx
import yaml

from src.utils.hash import stable_hash


LOW_RESOURCE_LANGS = ["bn", "sw", "te", "ne", "hi"]
RAG_METHODS = ["trag", "dkm_rag", "qtt_rag"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def eval_key(row: dict[str, Any]) -> str:
    dataset = str(row.get("_dataset") or row.get("dataset") or "")
    language = str(row.get("_language") or row.get("language") or "")
    sample_id = row.get("sample_id")
    question_id = row.get("question_id")
    if sample_id not in (None, ""):
        item_id = f"sample:{sample_id}"
    elif question_id not in (None, ""):
        item_id = f"question:{question_id}"
    else:
        item_id = f"content:{row.get('subject')}:{row.get('question')}"
    return f"{dataset}|{language}|{item_id}"


def option_labels(options: dict[str, Any]) -> list[str]:
    return sorted(str(key).upper() for key in options if re.fullmatch(r"[A-J]", str(key).upper()))


def format_options(options: dict[str, Any]) -> str:
    return "\n".join(f"{label}. {str(options[label]).strip()}" for label in option_labels(options))


def retrieval_query(row: dict[str, Any]) -> str:
    return f"{row['question']}\n{format_options(row['options'])}"


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    candidates = [text]
    candidates.extend(match.group(1).strip() for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_answer(text: str, labels: list[str]) -> tuple[str | None, str]:
    allowed = "".join(labels)
    parsed = parse_json_object(text)
    if parsed:
        answer = parsed.get("answer")
        if isinstance(answer, str) and answer.strip().upper() in labels:
            return answer.strip().upper(), "json"

    patterns = [
        rf"Final\s*answer\s*[:：]\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"Answer\s*[:：]\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"答案\s*[:：]\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"উত্তর\s*[:：]?\s*[\(\[]?\s*([{allowed}])\s*[\)\]]?",
        rf"^\s*([{allowed}])\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text or "", flags=re.I)
        if matches:
            return matches[-1].upper(), "regex"
    matches = re.findall(rf"(?<![A-Z])([{allowed}])(?![A-Z])", (text or "").upper())
    if matches:
        return matches[-1].upper(), "standalone_letter"
    return None, "unparsed"


def normalize_scores(value: Any, count: int) -> list[dict[str, int]]:
    if not isinstance(value, list):
        value = []
    normalized: list[dict[str, int]] = []
    for idx in range(count):
        raw = value[idx] if idx < len(value) and isinstance(value[idx], dict) else {}
        item: dict[str, int] = {}
        for key in ["semantic_equivalence", "grammatical_accuracy", "naturalness_fluency"]:
            try:
                score = int(raw.get(key, 0))
            except Exception:
                score = 0
            item[key] = max(0, min(5, score))
        normalized.append(item)
    return normalized


def normalize_text_list(value: Any, count: int, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        value = []
    output: list[str] = []
    for idx in range(count):
        item = value[idx] if idx < len(value) else None
        if isinstance(item, str) and item.strip():
            output.append(item.strip())
        else:
            output.append(fallback[idx] if idx < len(fallback) else "")
    return output


def load_eval_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    import random

    data_cfg = config["data"]
    languages = list(data_cfg.get("languages") or LOW_RESOURCE_LANGS)
    datasets = list(data_cfg.get("datasets") or ["global_mmlu", "mmlu_prox"])
    seed = int(config.get("seed", 42))
    limit_per_language = config.get("limit_per_language")
    rng = random.Random(seed)

    rows_by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dataset in datasets:
        for language in languages:
            if dataset == "global_mmlu":
                path = Path(data_cfg["global_mmlu_dir"]) / f"{language}.test.jsonl"
            elif dataset == "mmlu_prox":
                path = Path(data_cfg["mmlu_prox_dir"]) / f"{language}.test.jsonl"
            else:
                raise ValueError(f"unknown dataset: {dataset}")
            loaded = read_jsonl(path)
            for row in loaded:
                item = dict(row)
                item["_dataset"] = dataset
                item["_language"] = language
                item["eval_key"] = eval_key(item)
                rows_by_language[language].append(item)

    selected: list[dict[str, Any]] = []
    for language in sorted(rows_by_language):
        items = list(rows_by_language[language])
        rng.shuffle(items)
        if limit_per_language is not None:
            items = items[: int(limit_per_language)]
        selected.extend(items)
    selected.sort(key=lambda row: (row["_language"], row["_dataset"], str(row.get("sample_id") or row.get("question_id") or "")))
    return selected


class JsonCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, namespace: str, payload: Any) -> Path:
        return self.root / f"{namespace}-{stable_hash(payload, 32)}.json"

    def get(self, namespace: str, payload: Any) -> dict[str, Any] | None:
        path = self.path_for(namespace, payload)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, namespace: str, payload: Any, value: dict[str, Any]) -> None:
        path = self.path_for(namespace, payload)
        path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


class PassageStore:
    def __init__(self, *, db_path: Path, passages_dir: Path) -> None:
        self.con = sqlite3.connect(str(db_path), check_same_thread=False)
        self.passages_dir = passages_dir
        self.file_map = {int(idx): str(path) for idx, path in self.con.execute("select idx, path from files")}
        self.handles: dict[int, Any] = {}
        self.lock = threading.RLock()

    def fetch(self, passage_id: str) -> dict[str, Any]:
        with self.lock:
            row = self.con.execute("select f, off from pos where id = ?", (passage_id,)).fetchone()
            if row is None:
                return {"id": passage_id, "missing": "offset_not_found"}
            file_idx, offset = int(row[0]), int(row[1])
            handle = self.handles.get(file_idx)
            if handle is None:
                handle = (self.passages_dir / self.file_map[file_idx]).open("rb")
                self.handles[file_idx] = handle
            handle.seek(offset)
            line = handle.readline()
        return json.loads(line)


class EnglishWikiHttpRetriever:
    def __init__(self, *, base_url: str, timeout_s: float, cache: JsonCache) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.cache = cache
        self.client = httpx.Client(timeout=timeout_s)

    def retrieve(self, query: str, top_k: int) -> dict[str, Any]:
        payload = {"query": query, "top_k": int(top_k)}
        cached = self.cache.get("english_retrieval", payload)
        if cached is not None:
            return cached
        started = time.perf_counter()
        response = self.client.post(f"{self.base_url}/search", json=payload)
        response.raise_for_status()
        data = response.json()
        result = {
            "query": query,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "results": [
                {
                    "rank": item.get("rank"),
                    "score": item.get("score"),
                    "language": "en",
                    "passage_id": item.get("passage_id"),
                    "title": item.get("title"),
                    "section": item.get("section"),
                    "text": item.get("text") or "",
                }
                for item in data.get("results", [])
            ],
        }
        self.cache.set("english_retrieval", payload, result)
        return result


class MultilingualFaissRetriever:
    def __init__(
        self,
        *,
        root: Path,
        languages: list[str],
        model_dir: Path,
        device: str,
        cache: JsonCache,
        ef_search: int = 128,
    ) -> None:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.faiss = faiss
        self.np = np
        self.cache = cache
        self.model = SentenceTransformer(str(model_dir), device=device)
        self.shards: dict[str, dict[str, Any]] = {}
        for language in languages:
            lang_root = root / f"20231101.{language}"
            faiss_dir = lang_root / "faiss"
            index = faiss.read_index(str(faiss_dir / "hnsw.faiss"))
            if hasattr(index, "hnsw"):
                index.hnsw.efSearch = int(ef_search)
            ids = [line.rstrip("\n") for line in (faiss_dir / "ids.txt").open(encoding="utf-8")]
            self.shards[language] = {
                "index": index,
                "ids": ids,
                "store": PassageStore(db_path=faiss_dir / "offsets.sqlite", passages_dir=lang_root / "passages"),
            }
        dims = {int(shard["index"].d) for shard in self.shards.values()}
        if len(dims) != 1:
            raise ValueError(f"mixed multilingual index dimensions: {dims}")
        self.dim = dims.pop()

    def _encode(self, query: str) -> Any:
        vector = self.model.encode(
            [query],
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
            truncate_dim=self.dim,
        )
        vector = self.np.asarray(vector, dtype=self.np.float32)
        if vector.shape != (1, self.dim):
            raise ValueError(f"embedding shape {vector.shape} != (1, {self.dim})")
        return vector

    def retrieve(self, query: str, top_k: int) -> dict[str, Any]:
        payload = {"query": query, "top_k": int(top_k), "languages": sorted(self.shards)}
        cached = self.cache.get("multilingual_retrieval", payload)
        if cached is not None:
            return cached
        started = time.perf_counter()
        query_vec = self._encode(query)
        candidates: list[dict[str, Any]] = []
        for language, shard in self.shards.items():
            scores, hits = shard["index"].search(query_vec, int(top_k))
            for score, hit in zip(scores[0].tolist(), hits[0].tolist()):
                if hit < 0:
                    continue
                passage_id = shard["ids"][int(hit)]
                passage = shard["store"].fetch(passage_id)
                candidates.append(
                    {
                        "score": float(score),
                        "language": language,
                        "passage_id": passage_id,
                        "title": passage.get("title"),
                        "section": passage.get("section"),
                        "text": passage.get("text") or "",
                    }
                )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        results = []
        for rank, item in enumerate(candidates[: int(top_k)], start=1):
            item = dict(item)
            item["rank"] = rank
            results.append(item)
        result = {"query": query, "latency_ms": int((time.perf_counter() - started) * 1000), "results": results}
        self.cache.set("multilingual_retrieval", payload, result)
        return result


@dataclass
class LLMResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int
    finish_reason: str | None
    cache_hit: bool


class QwenVLLMChat:
    def __init__(
        self,
        *,
        model_dir: Path,
        model_name: str,
        cache: JsonCache,
        max_model_len: int,
        gpu_memory_utilization: float,
        batch_size: int,
        temperature: float,
    ) -> None:
        from transformers import AutoTokenizer

        self.model_dir = model_dir
        self.model_name = model_name
        self.cache = cache
        self.max_model_len = int(max_model_len)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.batch_size = int(batch_size)
        self.temperature = float(temperature)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self._llm: Any | None = None

    def _ensure_llm(self) -> Any:
        if self._llm is None:
            from vllm import LLM

            self._llm = LLM(
                model=str(self.model_dir),
                tokenizer=str(self.model_dir),
                trust_remote_code=True,
                dtype="bfloat16",
                tensor_parallel_size=1,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enable_prefix_caching=True,
            )
        return self._llm

    def render(self, messages: list[dict[str, str]]) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text or "", add_special_tokens=False))

    def truncate_tokens(self, text: str, max_tokens: int) -> str:
        ids = self.tokenizer.encode(text or "", add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text or ""
        return self.tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

    def generate_many(self, *, stage: str, messages_list: list[list[dict[str, str]]], max_tokens: int) -> list[LLMResult]:
        from vllm import SamplingParams

        results: list[LLMResult | None] = [None] * len(messages_list)
        missing: list[tuple[int, list[dict[str, str]], str, Path]] = []
        for idx, messages in enumerate(messages_list):
            prompt = self.render(messages)
            payload = {
                "stage": stage,
                "model": self.model_name,
                "messages": messages,
                "max_tokens": int(max_tokens),
                "temperature": self.temperature,
            }
            path = self.cache.path_for("llm", payload)
            if path.exists():
                cached = json.loads(path.read_text(encoding="utf-8"))
                results[idx] = LLMResult(cache_hit=True, **cached)
            else:
                missing.append((idx, messages, prompt, path))
        if missing:
            llm = self._ensure_llm()
            sampling = SamplingParams(temperature=self.temperature, max_tokens=int(max_tokens))
            for start in range(0, len(missing), self.batch_size):
                chunk = missing[start : start + self.batch_size]
                prompts = [item[2] for item in chunk]
                started = time.perf_counter()
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                per_item_latency = int(elapsed_ms / max(1, len(outputs)))
                for (idx, _messages, prompt, path), output in zip(chunk, outputs):
                    generated = output.outputs[0] if output.outputs else None
                    content = generated.text if generated else ""
                    completion_tokens = len(generated.token_ids) if generated else 0
                    prompt_tokens = self.token_count(prompt)
                    payload = {
                        "content": content,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "latency_ms": per_item_latency,
                        "finish_reason": getattr(generated, "finish_reason", None) if generated else None,
                    }
                    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
                    results[idx] = LLMResult(cache_hit=False, **payload)
        return [item for item in results if item is not None]


def system(content: str) -> dict[str, str]:
    return {"role": "system", "content": content}


def user(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def translation_messages(row: dict[str, Any], target_language: str) -> list[dict[str, str]]:
    return [
        system("You are a precise translator for multilingual exam questions. Do not solve the question."),
        user(
            "Translate the question and all options to "
            f"{target_language}. Return only JSON with keys question and options.\n\n"
            + json.dumps({"question": row["question"], "options": row["options"]}, ensure_ascii=False)
        ),
    ]


def passage_translation_messages(passages: list[dict[str, Any]], target_language: str) -> list[dict[str, str]]:
    compact = [
        {
            "id": idx,
            "language": passage.get("language"),
            "title": passage.get("title"),
            "text": (passage.get("text") or "")[:1600],
        }
        for idx, passage in enumerate(passages)
    ]
    return [
        system("You translate retrieved Wikipedia passages. Preserve facts and numbers. Do not add explanations."),
        user(
            f"Translate each passage into {target_language}. Return only JSON: "
            '{"translations":["..."]}. Keep the same order.\n\n'
            + json.dumps(compact, ensure_ascii=False)
        ),
    ]


def refine_messages(row: dict[str, Any], translated_passages: list[str]) -> list[dict[str, str]]:
    compact = {"question": row["question"], "options": row["options"], "translated_passages": translated_passages}
    return [
        system("You refine retrieved evidence for relevance. Preserve source meaning; do not add new facts."),
        user(
            "For each translated passage, rewrite a concise relevance-focused version for answering the question. "
            "If a passage is irrelevant, say so briefly. Return only JSON: {\"refined\":[\"...\"]}.\n\n"
            + json.dumps(compact, ensure_ascii=False)
        ),
    ]


def score_messages(row: dict[str, Any], translated_passages: list[str]) -> list[dict[str, str]]:
    compact = {"question": row["question"], "options": row["options"], "translated_passages": translated_passages}
    return [
        system("You are a translation quality judge. Score conservatively from 0 to 5."),
        user(
            "For each translated passage, score semantic_equivalence, grammatical_accuracy, and naturalness_fluency. "
            "Return only JSON: {\"scores\":[{\"semantic_equivalence\":0,\"grammatical_accuracy\":0,\"naturalness_fluency\":0}]}.\n\n"
            + json.dumps(compact, ensure_ascii=False)
        ),
    ]


def answer_messages(row: dict[str, Any], *, method: str, evidence_context: str, extra_context: str = "") -> list[dict[str, str]]:
    labels = "".join(option_labels(row["options"]))
    method_note = {
        "trag": "Use the original question, English translation, and English Wikipedia evidence.",
        "dkm_rag": "Use translated passages and refined relevance notes. If they conflict, trust the translated passage.",
        "qtt_rag": "Use tagged passages and prefer evidence with high semantic_equivalence.",
    }[method]
    return [
        system("You are a careful multilingual multiple-choice exam solver. Return only the answer label."),
        user(
            f"{method_note}\n\n"
            f"Question:\n{row['question']}\n\nOptions:\n{format_options(row['options'])}\n\n"
            f"{extra_context}\n\nEvidence:\n{evidence_context}\n\n"
            f"Choose the single best answer. Return exactly one capital letter from {labels}."
        ),
    ]


def translated_question_from_response(row: dict[str, Any], result: LLMResult) -> dict[str, Any]:
    parsed = parse_json_object(result.content) or {}
    question = parsed.get("question")
    options = parsed.get("options")
    if not isinstance(question, str) or not question.strip():
        question = result.content.strip() or row["question"]
    if not isinstance(options, dict):
        options = row["options"]
    return {"question": question, "options": options, "raw": result.content}


def text_for_docs(docs: list[dict[str, Any]]) -> list[str]:
    return [str(doc.get("text") or "") for doc in docs]


def build_evidence_context(
    *,
    method: str,
    docs: list[dict[str, Any]],
    tokenizer: QwenVLLMChat,
    max_evidence_tokens: int,
    per_doc_max_tokens: int,
) -> tuple[str, int]:
    chunks: list[str] = []
    for doc in docs:
        current = "\n\n".join(chunks)
        if tokenizer.token_count(current) >= int(max_evidence_tokens):
            break
        if method == "trag":
            body = f"[{doc.get('rank')}] {doc.get('title')} / {doc.get('section') or '<lead>'}\n{doc.get('text') or ''}"
        elif method == "dkm_rag":
            body = (
                f"[{doc.get('rank')}] {doc.get('title')} / {doc.get('section') or '<lead>'} ({doc.get('language')})\n"
                f"Translated passage:\n{doc.get('translated_text') or ''}\n"
                f"Refined relevance notes:\n{doc.get('refined_text') or ''}"
            )
        else:
            scores = doc.get("quality_scores") or {}
            body = (
                f"[{doc.get('rank')}] {doc.get('title')} / {doc.get('section') or '<lead>'} ({doc.get('language')})\n"
                "Quality tags: "
                f"semantic_equivalence={scores.get('semantic_equivalence', 0)}, "
                f"grammatical_accuracy={scores.get('grammatical_accuracy', 0)}, "
                f"naturalness_fluency={scores.get('naturalness_fluency', 0)}\n"
                f"Translated passage:\n{doc.get('translated_text') or doc.get('text') or ''}"
            )
        truncated = tokenizer.truncate_tokens(body, int(per_doc_max_tokens))
        candidate = "\n\n".join([*chunks, truncated])
        if tokenizer.token_count(candidate) <= int(max_evidence_tokens):
            chunks.append(truncated)
            continue
        remaining = int(max_evidence_tokens) - tokenizer.token_count(current)
        if chunks:
            # Reserve a small margin for the separator between passages.
            remaining = max(0, remaining - 4)
        if remaining <= 0:
            break
        while remaining > 0:
            truncated = tokenizer.truncate_tokens(body, remaining)
            candidate = "\n\n".join([*chunks, truncated])
            if tokenizer.token_count(candidate) <= int(max_evidence_tokens):
                chunks.append(truncated)
                break
            remaining -= 1
    context = "\n\n".join(chunks)
    return context, tokenizer.token_count(context)


def load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    items: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        items[str(row["eval_key"])] = row
    return items


def reset_method_outputs(output_dir: Path, method: str) -> None:
    """Clear resumable stage files for one method before a fresh non-resume run."""

    candidates = [
        output_dir / "retrieval" / f"{method}.jsonl",
        output_dir / "translations" / f"{method}.jsonl",
        output_dir / "dkm_refined" / f"{method}.jsonl",
        output_dir / "qtt_tags" / f"{method}.jsonl",
        output_dir / "predictions" / f"{method}.jsonl",
        output_dir / "token_logs" / f"{method}.jsonl",
    ]
    for path in candidates:
        if path.exists():
            path.unlink()


def stage_token_summary(results: list[LLMResult]) -> dict[str, int]:
    return {
        "prompt_tokens": sum(item.prompt_tokens for item in results),
        "completion_tokens": sum(item.completion_tokens for item in results),
        "total_tokens": sum(item.total_tokens for item in results),
        "latency_ms": sum(item.latency_ms for item in results),
    }


def build_prediction_record(
    *,
    row: dict[str, Any],
    method: str,
    raw_answer: str,
    answer_result: LLMResult,
    stage_results: dict[str, list[LLMResult]],
    retrieved_docs: list[dict[str, Any]],
    evidence_tokens: int,
    retrieval_latency_ms: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = option_labels(row["options"])
    answer, parse_source = parse_answer(raw_answer, labels)
    gold = str(row.get("answer") or row.get("answer_label") or "").strip().upper()
    stage_counts = {stage: stage_token_summary(results) for stage, results in stage_results.items()}
    total_prompt = sum(counts["prompt_tokens"] for counts in stage_counts.values())
    total_completion = sum(counts["completion_tokens"] for counts in stage_counts.values())
    total_latency = retrieval_latency_ms + sum(counts["latency_ms"] for counts in stage_counts.values())
    prediction = {
        "eval_key": row["eval_key"],
        "dataset": row["_dataset"],
        "language": row["_language"],
        "subject": row.get("subject"),
        "sample_id": row.get("sample_id"),
        "question_id": row.get("question_id"),
        "method": method,
        "answer": answer,
        "gold": gold,
        "correct": bool(answer == gold),
        "parse_success": bool(answer),
        "parse_source": parse_source,
        "raw_output": raw_answer,
        "finish_reason": answer_result.finish_reason,
    }
    token_log = {
        "eval_key": row["eval_key"],
        "method": method,
        "num_llm_calls": sum(len(results) for results in stage_results.values()),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "stage_level_token_counts": stage_counts,
        "retrieved_docs": len(retrieved_docs),
        "evidence_tokens_after_truncation": int(evidence_tokens),
        "latency_ms": int(total_latency),
        "answer": answer,
        "parse_success": bool(answer),
    }
    return prediction, token_log


def run_trag(
    *,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
    llm: QwenVLLMChat,
    english_retriever: EnglishWikiHttpRetriever,
) -> None:
    method = "trag"
    if not config.get("resume", True):
        reset_method_outputs(output_dir, method)
    pred_path = output_dir / "predictions" / f"{method}.jsonl"
    existing = load_existing_predictions(pred_path) if config.get("resume", True) else {}
    rows = [row for row in rows if row["eval_key"] not in existing]
    if not rows:
        return
    rag_cfg = config["rag"]
    target_messages = [translation_messages(row, "English") for row in rows]
    translation_results = llm.generate_many(stage="trag_translate_query", messages_list=target_messages, max_tokens=rag_cfg["translation_max_tokens"])
    translations = [translated_question_from_response(row, result) for row, result in zip(rows, translation_results)]
    append_jsonl(
        output_dir / "translations" / f"{method}.jsonl",
        [{"eval_key": row["eval_key"], "method": method, "translation": translation} for row, translation in zip(rows, translations)],
    )

    retrieval_records: list[dict[str, Any]] = []
    docs_by_key: dict[str, list[dict[str, Any]]] = {}
    retrieval_latency: dict[str, int] = {}
    for row, translation in zip(rows, translations):
        query = f"{translation['question']}\n{format_options(translation['options'])}"
        retrieval = english_retriever.retrieve(query, rag_cfg["top_k"])
        docs_by_key[row["eval_key"]] = retrieval["results"]
        retrieval_latency[row["eval_key"]] = int(retrieval.get("latency_ms") or 0)
        retrieval_records.append({"eval_key": row["eval_key"], "method": method, **retrieval})
    append_jsonl(output_dir / "retrieval" / f"{method}.jsonl", retrieval_records)

    contexts: list[tuple[str, int]] = []
    answer_prompts: list[list[dict[str, str]]] = []
    for row, translation in zip(rows, translations):
        context, evidence_tokens = build_evidence_context(
            method=method,
            docs=docs_by_key[row["eval_key"]],
            tokenizer=llm,
            max_evidence_tokens=rag_cfg["max_evidence_tokens"],
            per_doc_max_tokens=rag_cfg["per_doc_max_tokens"],
        )
        contexts.append((context, evidence_tokens))
        extra = "English translation:\n" + json.dumps({"question": translation["question"], "options": translation["options"]}, ensure_ascii=False)
        answer_prompts.append(answer_messages(row, method=method, evidence_context=context, extra_context=extra))
    answer_results = llm.generate_many(stage="trag_answer", messages_list=answer_prompts, max_tokens=rag_cfg["max_final_answer_tokens"])

    predictions: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    for idx, (row, answer_result, (_context, evidence_tokens)) in enumerate(zip(rows, answer_results, contexts)):
        prediction, token_log = build_prediction_record(
            row=row,
            method=method,
            raw_answer=answer_result.content,
            answer_result=answer_result,
            stage_results={"translate_query": [translation_results[idx]], "answer": [answer_result]},
            retrieved_docs=docs_by_key[row["eval_key"]],
            evidence_tokens=evidence_tokens,
            retrieval_latency_ms=retrieval_latency[row["eval_key"]],
        )
        predictions.append(prediction)
        logs.append(token_log)
    append_jsonl(pred_path, predictions)
    append_jsonl(output_dir / "token_logs" / f"{method}.jsonl", logs)


def run_dkm_rag(
    *,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
    llm: QwenVLLMChat,
    multilingual_retriever: MultilingualFaissRetriever,
) -> None:
    method = "dkm_rag"
    if not config.get("resume", True):
        reset_method_outputs(output_dir, method)
    pred_path = output_dir / "predictions" / f"{method}.jsonl"
    existing = load_existing_predictions(pred_path) if config.get("resume", True) else {}
    rows = [row for row in rows if row["eval_key"] not in existing]
    if not rows:
        return
    rag_cfg = config["rag"]
    docs_by_key: dict[str, list[dict[str, Any]]] = {}
    retrieval_latency: dict[str, int] = {}
    retrieval_records: list[dict[str, Any]] = []
    for row in rows:
        retrieval = multilingual_retriever.retrieve(retrieval_query(row), rag_cfg["top_k"])
        docs_by_key[row["eval_key"]] = retrieval["results"]
        retrieval_latency[row["eval_key"]] = int(retrieval.get("latency_ms") or 0)
        retrieval_records.append({"eval_key": row["eval_key"], "method": method, **retrieval})
    append_jsonl(output_dir / "retrieval" / f"{method}.jsonl", retrieval_records)

    translate_prompts = [passage_translation_messages(docs_by_key[row["eval_key"]], row["_language"]) for row in rows]
    translation_results = llm.generate_many(stage="dkm_translate_passages", messages_list=translate_prompts, max_tokens=rag_cfg["translation_max_tokens"])
    translated_lists: list[list[str]] = []
    for row, result in zip(rows, translation_results):
        docs = docs_by_key[row["eval_key"]]
        parsed = parse_json_object(result.content) or {}
        translated = normalize_text_list(parsed.get("translations"), len(docs), text_for_docs(docs))
        translated_lists.append(translated)
        for doc, translated_text in zip(docs, translated):
            doc["translated_text"] = translated_text
    append_jsonl(
        output_dir / "translations" / f"{method}.jsonl",
        [{"eval_key": row["eval_key"], "method": method, "translations": texts} for row, texts in zip(rows, translated_lists)],
    )

    refine_prompts = [refine_messages(row, texts) for row, texts in zip(rows, translated_lists)]
    refine_results = llm.generate_many(stage="dkm_refine_passages", messages_list=refine_prompts, max_tokens=rag_cfg["refine_max_tokens"])
    refined_lists: list[list[str]] = []
    for row, result in zip(rows, refine_results):
        docs = docs_by_key[row["eval_key"]]
        parsed = parse_json_object(result.content) or {}
        refined = normalize_text_list(parsed.get("refined"), len(docs), ["" for _ in docs])
        refined_lists.append(refined)
        for doc, refined_text in zip(docs, refined):
            doc["refined_text"] = refined_text
    append_jsonl(
        output_dir / "dkm_refined" / f"{method}.jsonl",
        [{"eval_key": row["eval_key"], "method": method, "refined": texts} for row, texts in zip(rows, refined_lists)],
    )

    contexts: list[tuple[str, int]] = []
    answer_prompts: list[list[dict[str, str]]] = []
    for row in rows:
        context, evidence_tokens = build_evidence_context(
            method=method,
            docs=docs_by_key[row["eval_key"]],
            tokenizer=llm,
            max_evidence_tokens=rag_cfg["max_evidence_tokens"],
            per_doc_max_tokens=rag_cfg["per_doc_max_tokens"],
        )
        contexts.append((context, evidence_tokens))
        answer_prompts.append(answer_messages(row, method=method, evidence_context=context))
    answer_results = llm.generate_many(stage="dkm_answer", messages_list=answer_prompts, max_tokens=rag_cfg["max_final_answer_tokens"])

    predictions: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    for idx, (row, answer_result, (_context, evidence_tokens)) in enumerate(zip(rows, answer_results, contexts)):
        prediction, token_log = build_prediction_record(
            row=row,
            method=method,
            raw_answer=answer_result.content,
            answer_result=answer_result,
            stage_results={
                "translate_passages": [translation_results[idx]],
                "refine_passages": [refine_results[idx]],
                "answer": [answer_result],
            },
            retrieved_docs=docs_by_key[row["eval_key"]],
            evidence_tokens=evidence_tokens,
            retrieval_latency_ms=retrieval_latency[row["eval_key"]],
        )
        predictions.append(prediction)
        logs.append(token_log)
    append_jsonl(pred_path, predictions)
    append_jsonl(output_dir / "token_logs" / f"{method}.jsonl", logs)


def run_qtt_rag(
    *,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
    llm: QwenVLLMChat,
    multilingual_retriever: MultilingualFaissRetriever,
) -> None:
    method = "qtt_rag"
    if not config.get("resume", True):
        reset_method_outputs(output_dir, method)
    pred_path = output_dir / "predictions" / f"{method}.jsonl"
    existing = load_existing_predictions(pred_path) if config.get("resume", True) else {}
    rows = [row for row in rows if row["eval_key"] not in existing]
    if not rows:
        return
    rag_cfg = config["rag"]
    docs_by_key: dict[str, list[dict[str, Any]]] = {}
    retrieval_latency: dict[str, int] = {}
    retrieval_records: list[dict[str, Any]] = []
    for row in rows:
        retrieval = multilingual_retriever.retrieve(retrieval_query(row), rag_cfg["top_k"])
        docs_by_key[row["eval_key"]] = retrieval["results"]
        retrieval_latency[row["eval_key"]] = int(retrieval.get("latency_ms") or 0)
        retrieval_records.append({"eval_key": row["eval_key"], "method": method, **retrieval})
    append_jsonl(output_dir / "retrieval" / f"{method}.jsonl", retrieval_records)

    translation_prompts: list[list[dict[str, str]]] = []
    translation_prompt_rows: list[int] = []
    non_query_positions_by_row: list[list[int]] = []
    for row in rows:
        docs = docs_by_key[row["eval_key"]]
        non_query_positions = [idx for idx, doc in enumerate(docs) if doc.get("language") != row["_language"]]
        non_query_positions_by_row.append(non_query_positions)
        docs_to_translate = [docs[idx] for idx in non_query_positions]
        if docs_to_translate:
            translation_prompt_rows.append(len(non_query_positions_by_row) - 1)
            translation_prompts.append(passage_translation_messages(docs_to_translate, row["_language"]))
    generated_translation_results = (
        llm.generate_many(stage="qtt_translate_non_query_passages", messages_list=translation_prompts, max_tokens=rag_cfg["translation_max_tokens"])
        if translation_prompts
        else []
    )
    translation_results: list[LLMResult | None] = [None for _ in rows]
    for row_idx, result in zip(translation_prompt_rows, generated_translation_results):
        translation_results[row_idx] = result

    translated_lists: list[list[str]] = []
    for row, result in zip(rows, translation_results):
        docs = docs_by_key[row["eval_key"]]
        translated = [doc.get("text") or "" for doc in docs]
        non_query_positions = non_query_positions_by_row[len(translated_lists)]
        if result is not None:
            parsed = parse_json_object(result.content) or {}
            non_query_translations = normalize_text_list(
                parsed.get("translations"),
                len(non_query_positions),
                [translated[idx] for idx in non_query_positions],
            )
            for idx, translated_text in zip(non_query_positions, non_query_translations):
                translated[idx] = translated_text
        translated_lists.append(translated)
        for doc, translated_text in zip(docs, translated):
            doc["translated_text"] = translated_text
    append_jsonl(
        output_dir / "translations" / f"{method}.jsonl",
        [{"eval_key": row["eval_key"], "method": method, "translations": texts} for row, texts in zip(rows, translated_lists)],
    )

    score_prompts = [score_messages(row, texts) for row, texts in zip(rows, translated_lists)]
    score_results = llm.generate_many(stage="qtt_score_quality", messages_list=score_prompts, max_tokens=rag_cfg["quality_score_max_tokens"])
    score_lists: list[list[dict[str, int]]] = []
    for row, result in zip(rows, score_results):
        docs = docs_by_key[row["eval_key"]]
        parsed = parse_json_object(result.content) or {}
        scores = normalize_scores(parsed.get("scores"), len(docs))
        score_lists.append(scores)
        for doc, score in zip(docs, scores):
            doc["quality_scores"] = score
    append_jsonl(
        output_dir / "qtt_tags" / f"{method}.jsonl",
        [{"eval_key": row["eval_key"], "method": method, "quality_scores": scores} for row, scores in zip(rows, score_lists)],
    )

    contexts: list[tuple[str, int]] = []
    answer_prompts: list[list[dict[str, str]]] = []
    for row in rows:
        context, evidence_tokens = build_evidence_context(
            method=method,
            docs=docs_by_key[row["eval_key"]],
            tokenizer=llm,
            max_evidence_tokens=rag_cfg["max_evidence_tokens"],
            per_doc_max_tokens=rag_cfg["per_doc_max_tokens"],
        )
        contexts.append((context, evidence_tokens))
        answer_prompts.append(answer_messages(row, method=method, evidence_context=context))
    answer_results = llm.generate_many(stage="qtt_answer", messages_list=answer_prompts, max_tokens=rag_cfg["max_final_answer_tokens"])

    predictions: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    for idx, (row, answer_result, (_context, evidence_tokens)) in enumerate(zip(rows, answer_results, contexts)):
        prediction, token_log = build_prediction_record(
            row=row,
            method=method,
            raw_answer=answer_result.content,
            answer_result=answer_result,
            stage_results={
                "translate_non_query_passages": [translation_results[idx]] if translation_results[idx] is not None else [],
                "quality_score": [score_results[idx]],
                "answer": [answer_result],
            },
            retrieved_docs=docs_by_key[row["eval_key"]],
            evidence_tokens=evidence_tokens,
            retrieval_latency_ms=retrieval_latency[row["eval_key"]],
        )
        predictions.append(prediction)
        logs.append(token_log)
    append_jsonl(pred_path, predictions)
    append_jsonl(output_dir / "token_logs" / f"{method}.jsonl", logs)


def summarize_outputs(output_dir: Path, methods: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"methods": {}}
    accuracy_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for method in methods:
        pred_path = output_dir / "predictions" / f"{method}.jsonl"
        log_path = output_dir / "token_logs" / f"{method}.jsonl"
        predictions = read_jsonl(pred_path) if pred_path.exists() else []
        logs = read_jsonl(log_path) if log_path.exists() else []
        by_language: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0, "parsed": 0})
        for row in predictions:
            bucket = by_language[row["language"]]
            bucket["n"] += 1
            bucket["correct"] += int(bool(row.get("correct")))
            bucket["parsed"] += int(bool(row.get("parse_success")))
        for language, bucket in sorted(by_language.items()):
            n = bucket["n"]
            accuracy_rows.append(
                {
                    "method": method,
                    "language": language,
                    "n": n,
                    "correct": bucket["correct"],
                    "accuracy": bucket["correct"] / n if n else 0.0,
                    "parse_rate": bucket["parsed"] / n if n else 0.0,
                }
            )
        n_pred = len(predictions)
        correct = sum(int(bool(row.get("correct"))) for row in predictions)
        parsed = sum(int(bool(row.get("parse_success"))) for row in predictions)
        denom = max(1, len(logs))
        cost = {
            "method": method,
            "avg_llm_calls": sum(float(row.get("num_llm_calls") or 0) for row in logs) / denom,
            "avg_prompt_tokens": sum(float(row.get("prompt_tokens") or 0) for row in logs) / denom,
            "avg_completion_tokens": sum(float(row.get("completion_tokens") or 0) for row in logs) / denom,
            "avg_total_tokens": sum(float(row.get("total_tokens") or 0) for row in logs) / denom,
            "avg_evidence_tokens": sum(float(row.get("evidence_tokens_after_truncation") or 0) for row in logs) / denom,
            "accuracy": correct / n_pred if n_pred else 0.0,
        }
        cost_rows.append(cost)
        summary["methods"][method] = {
            "n": n_pred,
            "correct": correct,
            "accuracy": correct / n_pred if n_pred else 0.0,
            "parsed": parsed,
            "parse_rate": parsed / n_pred if n_pred else 0.0,
            "by_language": by_language,
            "cost": cost,
        }

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "accuracy_by_language.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "language", "n", "correct", "accuracy", "parse_rate"])
        writer.writeheader()
        writer.writerows(accuracy_rows)
    with (metrics_dir / "cost_by_method.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "avg_llm_calls",
                "avg_prompt_tokens",
                "avg_completion_tokens",
                "avg_total_tokens",
                "avg_evidence_tokens",
                "accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(cost_rows)
    write_json(metrics_dir / "summary.json", summary)
    return summary


def prepare_output_dir(config_path: Path, config: dict[str, Any]) -> Path:
    output_root = Path(config["output_root"])
    run_name = str(config.get("run_name") or config.get("preset") or "smoke")
    output_dir = output_root / run_name
    for name in ["retrieval", "translations", "dkm_refined", "qtt_tags", "predictions", "token_logs", "metrics", "cache"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output_dir


def run_pipeline(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    output_dir = prepare_output_dir(config_path, config)
    rows = load_eval_rows(config)
    methods = list(config.get("methods") or RAG_METHODS)
    planned_calls_per_example = {"trag": 2, "dkm_rag": 3, "qtt_rag": 3}
    estimated_calls = sum(planned_calls_per_example[method] for method in methods) * len(rows)
    print(json.dumps({"planned_examples": len(rows), "methods": methods, "estimated_llm_calls": estimated_calls}, ensure_ascii=False), flush=True)

    cache = JsonCache(output_dir / "cache")
    llm_cfg = config["llm"]
    llm = QwenVLLMChat(
        model_dir=Path(llm_cfg["model_dir"]),
        model_name=llm_cfg.get("model_name", "Qwen3-8B-Instruct"),
        cache=cache,
        max_model_len=llm_cfg.get("max_model_len", 16384),
        gpu_memory_utilization=llm_cfg.get("gpu_memory_utilization", 0.70),
        batch_size=llm_cfg.get("batch_size", 64),
        temperature=config["rag"].get("temperature", 0.0),
    )

    retrieval_cfg = config["retrieval"]
    english_retriever = EnglishWikiHttpRetriever(
        base_url=retrieval_cfg.get("english_service_url", "http://127.0.0.1:8897"),
        timeout_s=float(retrieval_cfg.get("timeout_s", 60.0)),
        cache=cache,
    )
    multilingual_retriever: MultilingualFaissRetriever | None = None
    if any(method in methods for method in ["dkm_rag", "qtt_rag"]):
        multilingual_retriever = MultilingualFaissRetriever(
            root=Path(retrieval_cfg["multilingual_root"]),
            languages=list(config["data"].get("languages") or LOW_RESOURCE_LANGS),
            model_dir=Path(retrieval_cfg["embedding_model_dir"]),
            device=retrieval_cfg.get("device", "cuda"),
            cache=cache,
            ef_search=int(retrieval_cfg.get("ef_search", 128)),
        )

    if "trag" in methods:
        run_trag(rows=rows, config=config, output_dir=output_dir, llm=llm, english_retriever=english_retriever)
    if "dkm_rag" in methods:
        assert multilingual_retriever is not None
        run_dkm_rag(rows=rows, config=config, output_dir=output_dir, llm=llm, multilingual_retriever=multilingual_retriever)
    if "qtt_rag" in methods:
        assert multilingual_retriever is not None
        run_qtt_rag(rows=rows, config=config, output_dir=output_dir, llm=llm, multilingual_retriever=multilingual_retriever)

    summary = summarize_outputs(output_dir, methods)
    manifest = {
        "run_name": config.get("run_name"),
        "output_dir": str(output_dir),
        "methods": methods,
        "languages": config["data"].get("languages"),
        "datasets": config["data"].get("datasets"),
        "example_count": len(rows),
        "estimated_llm_calls": estimated_calls,
        "summary": summary,
        "finished_at": time.time(),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "manifest": str(output_dir / "manifest.json")}, ensure_ascii=False), flush=True)
    return manifest


def load_config(path: Path, preset: str | None = None) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if preset:
        presets = config.get("presets") or {}
        if preset not in presets:
            raise ValueError(f"unknown preset {preset}; available={sorted(presets)}")
        merged = dict(config)
        merged.update(presets[preset])
        merged["preset"] = preset
        config = merged
    config.pop("presets", None)
    return config
