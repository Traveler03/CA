from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.rag_methods import (
    LANGUAGE_NAMES,
    ProcessedPassage,
    RagMethod,
    WikiPassage,
    dkm_refine_batch_messages,
    dkm_refine_messages,
    english_query_from_translated_payload,
    mock_translated_question,
    normalize_quality_scores,
    option_labels,
    parse_json_object,
    qtt_quality_batch_messages,
    qtt_quality_messages,
    retrieval_query_text,
    solver_messages,
    supported_language,
    translate_passages_batch_messages,
    translate_passage_messages,
    translate_query_messages,
)
from src.clients.chat_client import AsyncChatClient
from src.runtime.resolve_codex_provider import resolve_provider
from src.utils.hash import stable_hash
from src.utils.jsonl import read_jsonl, write_jsonl


DEFAULT_CONFIG = Path("configs/rag_baseline_smoke.yaml")
SMOKE_MAX_ROWS = 100


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adapted DKM-RAG, QTT-RAG, and tRAG smoke baselines.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--methods", nargs="+", choices=["trag", "dkm_rag", "qtt_rag"])
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--global-mmlu-dir", type=Path)
    parser.add_argument("--languages", nargs="+")
    parser.add_argument("--limit-per-language", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--candidate-top-k", type=int)
    parser.add_argument("--final-top-k", type=int)
    parser.add_argument("--generation-top-k", type=int)
    parser.add_argument("--retrieval-mode", choices=["mock", "service", "faithful"])
    parser.add_argument("--retrieval-endpoint")
    parser.add_argument("--english-service-url")
    parser.add_argument("--multilingual-root", type=Path)
    parser.add_argument("--embedding-model-dir", type=Path)
    parser.add_argument("--embedding-device")
    parser.add_argument("--ef-search", type=int)
    parser.add_argument("--reranker", choices=["lexical", "none"])
    parser.add_argument("--live-model", action="store_true", help="Actually call the configured OpenAI-compatible model.")
    parser.add_argument("--local-vllm", action="store_true", help="Use a local vLLM model instead of the configured API model.")
    parser.add_argument("--local-model-dir", type=Path)
    parser.add_argument("--local-model-name")
    parser.add_argument("--local-max-model-len", type=int)
    parser.add_argument("--local-gpu-memory-utilization", type=float)
    parser.add_argument("--local-batch-size", type=int)
    parser.add_argument("--local-batch-timeout-ms", type=int)
    parser.add_argument("--batch-max-tokens", type=int)
    parser.add_argument("--max-context-chars-per-doc", type=int)
    parser.add_argument("--max-concurrent-model-requests", type=int)
    parser.add_argument("--max-model-calls", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--allow-full-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--qtt-keep-top-if-empty", action="store_true")
    return parser.parse_args(argv)


def merged_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    for key in [
        "output_dir",
        "methods",
        "input_jsonl",
        "global_mmlu_dir",
        "languages",
        "limit_per_language",
        "seed",
        "top_k",
        "candidate_top_k",
        "final_top_k",
        "generation_top_k",
        "retrieval_mode",
        "retrieval_endpoint",
        "english_service_url",
        "multilingual_root",
        "embedding_model_dir",
        "embedding_device",
        "ef_search",
        "reranker",
        "local_model_dir",
        "local_model_name",
        "local_max_model_len",
        "local_gpu_memory_utilization",
        "local_batch_size",
        "local_batch_timeout_ms",
        "batch_max_tokens",
        "max_context_chars_per_doc",
        "max_concurrent_model_requests",
        "max_model_calls",
        "max_rows",
    ]:
        arg_key = key.replace("-", "_")
        value = getattr(args, arg_key, None)
        if value is not None:
            config[key] = value
    config["live_model"] = bool(args.live_model or config.get("live_model", False))
    config["local_vllm"] = bool(args.local_vllm or config.get("local_vllm", False))
    config["allow_full_run"] = bool(args.allow_full_run or config.get("allow_full_run", False))
    config["resume"] = bool(args.resume or config.get("resume", False))
    config["dry_run_plan"] = bool(args.dry_run_plan)
    config["qtt_keep_top_if_empty"] = bool(args.qtt_keep_top_if_empty or config.get("qtt_keep_top_if_empty", False))
    return config


def load_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("input_jsonl"):
        rows = list(read_jsonl(Path(config["input_jsonl"])))
    else:
        base = Path(config.get("global_mmlu_dir") or "data/processed/global_mmlu")
        rows = []
        for language in config.get("languages", ["bn", "hi", "sw", "te", "ne"]):
            path = base / f"{language}.test.jsonl"
            lang_rows = list(read_jsonl(path))
            limit = config.get("limit_per_language")
            if limit is not None:
                lang_rows = lang_rows[: int(limit)]
            for row in lang_rows:
                row = dict(row)
                row.setdefault("language", language)
                row.setdefault("_dataset", "global_mmlu")
                rows.append(row)
    allow_full_run = bool(config.get("allow_full_run"))
    raw_max_rows = config.get("max_rows")
    max_rows = int(raw_max_rows) if raw_max_rows is not None else (None if allow_full_run else SMOKE_MAX_ROWS)
    if max_rows is not None and len(rows) > max_rows:
        rows = rows[:max_rows]
    if len(rows) > SMOKE_MAX_ROWS and not allow_full_run:
        raise RuntimeError(f"smoke guard: refusing to process {len(rows)} rows; max allowed is {SMOKE_MAX_ROWS}")
    return rows


def eval_key(method: str, row: dict[str, Any]) -> str:
    dataset = str(row.get("_dataset") or row.get("dataset") or "global_mmlu")
    language = str(row.get("language") or row.get("_language") or "")
    sample_id = str(row.get("sample_id") or row.get("question_id") or row.get("id") or row.get("question"))
    return f"{method}|{dataset}|{language}|{sample_id}"


def estimate_model_calls(rows: list[dict[str, Any]], methods: list[RagMethod], generation_top_k: int) -> dict[str, int]:
    per_method: dict[str, int] = {}
    for method in methods:
        # tRAG: query translation + final answer.
        calls_per_row = 2
        if method == "dkm_rag":
            # Batch faithful adapted upper bound: translate final non-query-language
            # passages together, refine final passages together, answer once.
            # Query is not translated.
            calls_per_row = 3
        elif method == "qtt_rag":
            # Batch faithful adapted upper bound: translate final non-query-language
            # passages together, score translations together, answer once.
            # Query is not translated.
            calls_per_row = 3
        per_method[method] = calls_per_row * len(rows)
    per_method["total"] = sum(per_method.values())
    return per_method


class MockModel:
    def __init__(self) -> None:
        self.calls = 0

    async def json(self, messages: list[dict[str, str]], *, namespace: str) -> dict[str, Any]:
        self.calls += 1
        if "translation quality" in messages[0]["content"].lower() or "quality-aware" in messages[0]["content"].lower():
            user = messages[-1]["content"]
            if '"scores"' in user or "Keep the same order" in user:
                count = max(1, user.count('"original_source_passage"'))
                return {
                    "scores": [
                        {
                            "semantic_equivalence": 5.0,
                            "grammatical_accuracy": 5.0,
                            "naturalness_fluency": 5.0,
                        }
                        for _ in range(count)
                    ]
                }
            return {
                "semantic_equivalence": 5.0,
                "grammatical_accuracy": 5.0,
                "naturalness_fluency": 5.0,
            }
        if "translate multilingual multiple-choice" in messages[0]["content"].lower():
            return {"question": "mock translated retrieval query", "options": {"A": "option A", "B": "option B", "C": "option C", "D": "option D"}}
        if "faithful document translator" in messages[0]["content"].lower():
            user = messages[-1]["content"]
            count = max(1, user.count('"source_language"'))
            return {"translations": ["mock translated passage" for _ in range(count)]}
        if "dual knowledge multilingual rag" in messages[0]["content"].lower():
            user = messages[-1]["content"]
            count = max(1, user.count('"translated_passages"'))
            try:
                payload = parse_json_object(user)
                passages = payload.get("translated_passages") if isinstance(payload, dict) else None
                if isinstance(passages, list):
                    count = len(passages)
            except Exception:
                pass
            return {"refined": ["mock refined passage" for _ in range(count)]}
        return {"answer": "A"}

    async def text(self, messages: list[dict[str, str]], *, namespace: str) -> str:
        self.calls += 1
        user = messages[-1]["content"]
        if "Translate the following English passage" in user:
            return "mock translated passage"
        if "Translated passage:" in user:
            return "mock refined passage"
        return "mock text"

    async def aclose(self) -> None:
        return None


class LiveModel:
    def __init__(self, *, output_dir: Path, concurrency: int) -> None:
        resolved, api_key = resolve_provider(".")
        self.client = AsyncChatClient(
            base_url=resolved.base_url,
            api_key=api_key,
            model=resolved.chat_model,
            concurrency=concurrency,
            timeout_s=120.0,
            max_retries=2,
            cache_dir=output_dir / "cache" / "chat",
            raw_dir=output_dir / "raw" / "chat",
            cache_enabled=True,
        )
        self.calls = 0

    async def json(self, messages: list[dict[str, str]], *, namespace: str) -> dict[str, Any]:
        self.calls += 1
        max_tokens = 512
        if any(marker in namespace for marker in ["translate_docs_batch", "refine_docs_batch", "quality_docs_batch"]):
            max_tokens = 4096
        result = await self.client.create(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            response_format={"type": "json_object"},
            retry_on_think=True,
        )
        if not result.ok:
            raise RuntimeError(result.error or result.invalid_reason or "model call failed")
        return parse_json_object(result.content)

    async def text(self, messages: list[dict[str, str]], *, namespace: str) -> str:
        self.calls += 1
        result = await self.client.create(messages, max_tokens=512, temperature=0.0, retry_on_think=True)
        if not result.ok:
            raise RuntimeError(result.error or result.invalid_reason or "model call failed")
        return result.content.strip()

    async def aclose(self) -> None:
        await self.client.aclose()


class LocalVLLMModel:
    def __init__(
        self,
        *,
        model_dir: Path,
        model_name: str = "Qwen3-8B",
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.70,
        batch_max_tokens: int = 4096,
        local_batch_size: int = 1,
        local_batch_timeout_ms: int = 0,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self.model_dir = model_dir
        self.model_name = model_name
        self.calls = 0
        self.batch_max_tokens = int(batch_max_tokens)
        self.local_batch_size = max(1, int(local_batch_size))
        self.local_batch_timeout_s = max(0.0, int(local_batch_timeout_ms) / 1000.0)
        self._lock = asyncio.Lock()
        self._batch_queue: asyncio.Queue[dict[str, Any]] | None = None
        self._batch_worker_task: asyncio.Task[None] | None = None
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.llm = LLM(
            model=str(model_dir),
            tokenizer=str(model_dir),
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=int(max_model_len),
            gpu_memory_utilization=float(gpu_memory_utilization),
            enable_prefix_caching=True,
        )
        if self.local_batch_size > 1:
            self._batch_queue = asyncio.Queue()
            self._batch_worker_task = asyncio.create_task(self._batch_worker())

    def _render(self, messages: list[dict[str, str]]) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _generate_once(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        from vllm import SamplingParams

        prompt = self._render(messages)
        sampling = SamplingParams(temperature=0.0, max_tokens=int(max_tokens))
        outputs = self.llm.generate([prompt], sampling, use_tqdm=False)
        if not outputs or not outputs[0].outputs:
            return ""
        return (outputs[0].outputs[0].text or "").strip()

    def _generate_many(self, messages_list: list[list[dict[str, str]]], *, max_tokens: int) -> list[str]:
        from vllm import SamplingParams

        prompts = [self._render(messages) for messages in messages_list]
        sampling = SamplingParams(temperature=0.0, max_tokens=int(max_tokens))
        outputs = self.llm.generate(prompts, sampling, use_tqdm=False)
        texts: list[str] = []
        for output in outputs:
            if not output.outputs:
                texts.append("")
            else:
                texts.append((output.outputs[0].text or "").strip())
        return texts

    async def _batch_worker(self) -> None:
        assert self._batch_queue is not None
        while True:
            first = await self._batch_queue.get()
            batch = [first]
            deferred: list[dict[str, Any]] = []
            max_tokens = int(first["max_tokens"])
            if self.local_batch_timeout_s > 0:
                await asyncio.sleep(self.local_batch_timeout_s)
            while len(batch) < self.local_batch_size:
                try:
                    item = self._batch_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if int(item["max_tokens"]) == max_tokens:
                    batch.append(item)
                else:
                    deferred.append(item)
            for item in deferred:
                self._batch_queue.put_nowait(item)
            try:
                texts = await asyncio.to_thread(
                    self._generate_many,
                    [item["messages"] for item in batch],
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                for item in batch:
                    if not item["future"].done():
                        item["future"].set_exception(exc)
            else:
                for item, text in zip(batch, texts):
                    if not item["future"].done():
                        item["future"].set_result(text)

    async def _generate(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        if self._batch_queue is None:
            async with self._lock:
                return await asyncio.to_thread(self._generate_once, messages, max_tokens=max_tokens)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        await self._batch_queue.put({"messages": messages, "max_tokens": int(max_tokens), "future": future})
        return await future

    async def json(self, messages: list[dict[str, str]], *, namespace: str) -> dict[str, Any]:
        self.calls += 1
        max_tokens = 512
        if any(marker in namespace for marker in ["translate_docs_batch", "refine_docs_batch", "quality_docs_batch"]):
            max_tokens = self.batch_max_tokens
        text = await self._generate(messages, max_tokens=max_tokens)
        return parse_json_object(text)

    async def text(self, messages: list[dict[str, str]], *, namespace: str) -> str:
        self.calls += 1
        return await self._generate(messages, max_tokens=512)

    async def aclose(self) -> None:
        if self._batch_worker_task is not None:
            self._batch_worker_task.cancel()
        return None


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

    def close(self) -> None:
        with self.lock:
            for handle in self.handles.values():
                handle.close()
            self.handles.clear()
            self.con.close()


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
            index_path = faiss_dir / "hnsw.faiss"
            ids_path = faiss_dir / "ids.txt"
            db_path = faiss_dir / "offsets.sqlite"
            passages_dir = lang_root / "passages"
            if not index_path.exists():
                raise FileNotFoundError(index_path)
            index = faiss.read_index(str(index_path))
            if hasattr(index, "hnsw"):
                index.hnsw.efSearch = int(ef_search)
            ids = [line.rstrip("\n") for line in ids_path.open(encoding="utf-8")]
            self.shards[language] = {
                "index": index,
                "ids": ids,
                "store": PassageStore(db_path=db_path, passages_dir=passages_dir),
            }
        dims = {int(shard["index"].d) for shard in self.shards.values()}
        if len(dims) != 1:
            raise ValueError(f"mixed multilingual index dimensions: {dims}")
        self.dim = dims.pop()

    def close(self) -> None:
        for shard in self.shards.values():
            shard["store"].close()

    def _encode(self, query: str) -> Any:
        kwargs: dict[str, Any] = {
            "prompt_name": "query",
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        try:
            vector = self.model.encode([query], truncate_dim=self.dim, **kwargs)
        except TypeError:
            vector = self.model.encode([query], **kwargs)
            if vector.shape[1] > self.dim:
                vector = vector[:, : self.dim]
                vector = vector / self.np.maximum(self.np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
        vector = self.np.asarray(vector, dtype=self.np.float32)
        if vector.shape != (1, self.dim):
            raise ValueError(f"embedding shape {vector.shape} != (1, {self.dim})")
        return vector

    def retrieve(self, query: str, *, top_k: int) -> list[WikiPassage]:
        payload = {"query": query, "top_k": int(top_k), "languages": sorted(self.shards)}
        cached = self.cache.get("multilingual_retrieval", payload)
        if cached is not None:
            return [passage_from_payload(item) for item in cached.get("results", [])]
        query_vec = self._encode(query)
        candidates: list[WikiPassage] = []
        per_shard_top_k = max(int(top_k), int(math.ceil(int(top_k) / max(1, len(self.shards)))))
        for language, shard in self.shards.items():
            scores, hits = shard["index"].search(query_vec, per_shard_top_k)
            for score, hit in zip(scores[0].tolist(), hits[0].tolist()):
                if hit < 0:
                    continue
                passage_id = shard["ids"][int(hit)]
                passage = shard["store"].fetch(passage_id)
                candidates.append(
                    WikiPassage(
                        passage_id=str(passage_id),
                        title=str(passage.get("title") or ""),
                        section=passage.get("section"),
                        score=float(score),
                        text=str(passage.get("text") or ""),
                        language=language,
                    )
                )
        candidates.sort(key=lambda item: item.score, reverse=True)
        results = [with_rank(item, rank) for rank, item in enumerate(candidates[: int(top_k)], start=1)]
        self.cache.set("multilingual_retrieval", payload, {"results": [passage_to_payload(item) for item in results]})
        return results


@dataclass
class RetrievalResources:
    cache: JsonCache
    multilingual: MultilingualFaissRetriever | None = None

    def close(self) -> None:
        if self.multilingual is not None:
            self.multilingual.close()


def with_rank(passage: WikiPassage, rank: int) -> WikiPassage:
    return WikiPassage(
        passage_id=passage.passage_id,
        title=passage.title,
        text=passage.text,
        score=passage.score,
        section=passage.section,
        language=passage.language,
        rank=rank,
    )


def passage_to_payload(passage: WikiPassage) -> dict[str, Any]:
    return {
        "passage_id": passage.passage_id,
        "title": passage.title,
        "section": passage.section,
        "score": passage.score,
        "text": passage.text,
        "language": passage.language,
        "rank": passage.rank,
    }


def passage_from_payload(item: dict[str, Any]) -> WikiPassage:
    return WikiPassage(
        passage_id=str(item.get("passage_id") or item.get("id") or ""),
        title=str(item.get("title") or ""),
        section=item.get("section"),
        score=float(item.get("score") or 0.0),
        text=str(item.get("text") or ""),
        language=str(item.get("language") or "en"),
        rank=int(item["rank"]) if item.get("rank") is not None else None,
    )


async def retrieve_english_passages(query: str, *, config: dict[str, Any], resources: RetrievalResources) -> list[WikiPassage]:
    top_k = int(config.get("candidate_top_k") or config.get("top_k") or 50)
    if config.get("retrieval_mode") == "mock":
        return [
            WikiPassage(
                passage_id=f"mock:en:{idx}",
                title=f"Mock English evidence {idx}",
                section=None,
                score=1.0 / idx,
                text=f"Mock English evidence. Query: {query}",
                language="en",
                rank=idx,
            )
            for idx in range(1, top_k + 1)
        ]
    endpoint = str(config.get("retrieval_endpoint") or "").strip()
    if not endpoint:
        base_url = str(config.get("english_service_url") or "http://127.0.0.1:8897").rstrip("/")
        endpoint = f"{base_url}/search"
    payload = {"query": query, "top_k": top_k}
    cached = resources.cache.get("english_retrieval", payload)
    if cached is not None:
        return [passage_from_payload(item) for item in cached.get("results", [])]
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
    passages = []
    for rank, item in enumerate(data.get("results", []), start=1):
        passages.append(
            WikiPassage(
                passage_id=str(item.get("passage_id") or item.get("id") or ""),
                title=str(item.get("title") or ""),
                section=item.get("section"),
                score=float(item.get("score") or 0.0),
                text=str(item.get("text") or ""),
                language="en",
                rank=int(item.get("rank") or rank),
            )
        )
    resources.cache.set("english_retrieval", payload, {"results": [passage_to_payload(item) for item in passages]})
    return passages


async def retrieve_multilingual_passages(
    query: str,
    *,
    config: dict[str, Any],
    row: dict[str, Any],
    resources: RetrievalResources,
) -> list[WikiPassage]:
    top_k = int(config.get("candidate_top_k") or config.get("top_k") or 50)
    if config.get("retrieval_mode") == "mock":
        query_language = supported_language(row)
        languages = [query_language, *[language for language in LANGUAGE_NAMES if language not in {query_language, "en"}]]
        subject = str(row.get("subject") or "unknown")
        return [
            WikiPassage(
                passage_id=f"mock:{languages[(idx - 1) % len(languages)]}:{subject}:{idx}",
                title=f"Mock multilingual evidence {idx}",
                section=None,
                score=1.0 / idx,
                text=f"Mock multilingual evidence for {subject}. Query: {query}",
                language=languages[(idx - 1) % len(languages)],
                rank=idx,
            )
            for idx in range(1, top_k + 1)
        ]
    if resources.multilingual is None:
        raise RuntimeError("multilingual retriever is not initialized")
    return resources.multilingual.retrieve(query, top_k=top_k)


_TOKEN_RE = re.compile(r"[\w\u0980-\u09FF\u0900-\u097F\u0C00-\u0C7F]+", flags=re.UNICODE)


def lexical_rerank(
    row: dict[str, Any],
    passages: list[WikiPassage],
    *,
    final_top_k: int,
    reranker: str,
    query_text: str | None = None,
) -> list[WikiPassage]:
    if reranker == "none":
        return [with_rank(passage, rank) for rank, passage in enumerate(passages[:final_top_k], start=1)]
    query_language = supported_language(row)
    query_tokens = set(token.lower() for token in _TOKEN_RE.findall(query_text or retrieval_query_text(row)))
    denom = max(1, len(query_tokens))

    def score_tuple(passage: WikiPassage) -> tuple[float, int, str]:
        text = f"{passage.title} {passage.section or ''} {passage.text}"
        passage_tokens = set(token.lower() for token in _TOKEN_RE.findall(text))
        lexical = len(query_tokens & passage_tokens) / denom
        language_bonus = 0.02 if passage.language == query_language else 0.0
        combined = float(passage.score) + lexical + language_bonus
        return combined, -int(passage.rank or 10**9), passage.passage_id

    reranked = sorted(passages, key=score_tuple, reverse=True)[:final_top_k]
    return [with_rank(passage, rank) for rank, passage in enumerate(reranked, start=1)]


async def process_passages(
    row: dict[str, Any],
    method: RagMethod,
    passages: list[WikiPassage],
    *,
    model: MockModel | LiveModel | LocalVLLMModel,
    config: dict[str, Any],
    row_key: str,
) -> list[ProcessedPassage]:
    language = supported_language(row)
    final_top_k = int(config.get("final_top_k") or config.get("generation_top_k") or min(5, len(passages)))
    selected = passages[:final_top_k]
    processed: list[ProcessedPassage] = []
    if method == "trag":
        return [ProcessedPassage(passage=passage) for passage in selected]
    translated_by_idx = [passage.text for passage in selected]
    non_query_indices = [idx for idx, passage in enumerate(selected) if passage.language != language]
    translation_failed = False
    if non_query_indices:
        docs_to_translate = [selected[idx] for idx in non_query_indices]
        try:
            raw_translation = await model.json(
                translate_passages_batch_messages(docs_to_translate, target_language=language),
                namespace=f"{row_key}.translate_docs_batch",
            )
        except Exception:
            translation_failed = True
            raw_translation = {"translations": [passage.text for passage in docs_to_translate]}
        translations = normalize_text_list(
            raw_translation.get("translations"),
            len(docs_to_translate),
            [passage.text for passage in docs_to_translate],
        )
        for idx, translated in zip(non_query_indices, translations):
            translated_by_idx[idx] = translated

    if method == "dkm_rag":
        refine_failed = False
        try:
            raw_refined = await model.json(
                dkm_refine_batch_messages(row, translated_by_idx),
                namespace=f"{row_key}.refine_docs_batch",
            )
        except Exception:
            refine_failed = True
            raw_refined = {"refined": ["" for _ in selected]}
        refined_by_idx = normalize_text_list(raw_refined.get("refined"), len(selected), ["" for _ in selected])
        for idx, passage in enumerate(selected):
            outcome = "original_query_language_refined" if passage.language == language else "translated_refined"
            if translation_failed and passage.language != language:
                outcome = "translation_fallback_refined"
            if refine_failed:
                outcome = outcome.replace("_refined", "_refine_fallback")
            processed.append(
                ProcessedPassage(
                    passage=passage,
                    translated_text=translated_by_idx[idx],
                    refined_text=refined_by_idx[idx],
                    keep=True,
                    outcome=outcome,
                )
            )
    else:
        score_by_idx: dict[int, dict[str, float]] = {}
        score_failed = False
        if non_query_indices:
            pairs = [
                {"original": selected[idx].text, "translated": translated_by_idx[idx]}
                for idx in non_query_indices
            ]
            try:
                raw_scores = await model.json(
                    qtt_quality_batch_messages(pairs, target_language=language),
                    namespace=f"{row_key}.quality_docs_batch",
                )
            except Exception:
                score_failed = True
                raw_scores = {"scores": [normalize_quality_scores({}) for _ in non_query_indices]}
            score_list = normalize_quality_score_list(raw_scores.get("scores"), len(non_query_indices))
            for idx, score in zip(non_query_indices, score_list):
                score_by_idx[idx] = score
        for idx, passage in enumerate(selected):
            if passage.language == language:
                processed.append(
                    ProcessedPassage(
                        passage=passage,
                        translated_text=None,
                        keep=True,
                        quality_scores={},
                        outcome="original_query_language",
                    )
                )
            else:
                processed.append(
                    ProcessedPassage(
                        passage=passage,
                        translated_text=translated_by_idx[idx],
                        keep=True,
                        quality_scores=score_by_idx.get(idx, normalize_quality_scores({})),
                        outcome=(
                            "translation_fallback_scored"
                            if translation_failed
                            else "translated_score_fallback"
                            if score_failed
                            else "translated_scored"
                        ),
                    )
                )
    # Keep the per-passage prompt functions imported for compatibility with old
    # cached runs and unit-level references. Main DKM/QTT smoke runs use the
    # batch path above to stay within AGENTS.md model-call limits.
    _ = (translate_passage_messages, dkm_refine_messages, qtt_quality_messages)
    return processed


def parse_answer_label(payload: dict[str, Any], row: dict[str, Any]) -> tuple[str | None, bool]:
    answer = str(payload.get("answer") or "").strip().upper()
    labels = set(option_labels(row.get("options") or {}))
    if answer in labels:
        return answer, True
    return None, False


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


def normalize_quality_score_list(value: Any, count: int) -> list[dict[str, float]]:
    if not isinstance(value, list):
        value = []
    output: list[dict[str, float]] = []
    for idx in range(count):
        item = value[idx] if idx < len(value) and isinstance(value[idx], dict) else {}
        output.append(normalize_quality_scores(item))
    return output


def truncate_context_text(text: str, *, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


async def run_one(
    row: dict[str, Any],
    method: RagMethod,
    *,
    model: MockModel | LiveModel | LocalVLLMModel,
    config: dict[str, Any],
    resources: RetrievalResources,
) -> dict[str, Any]:
    started = time.perf_counter()
    key = eval_key(method, row)
    candidate_top_k = int(config.get("candidate_top_k") or config.get("top_k") or 50)
    final_top_k = int(config.get("final_top_k") or config.get("generation_top_k") or 5)
    reranker = str(config.get("reranker") or "lexical")
    translated_query_payload: dict[str, Any] | None = None
    if method == "trag":
        try:
            translated_query_payload = await model.json(translate_query_messages(row), namespace=f"{key}.translate_query")
        except Exception as exc:
            translated_query_payload = mock_translated_question(row)
            translated_query_payload["_fallback_reason"] = f"translate_query_failed: {type(exc).__name__}"
        if not translated_query_payload.get("question"):
            translated_query_payload = mock_translated_question(row)
        retrieval_query = english_query_from_translated_payload(translated_query_payload)
        retrieval_scope = "english_only"
        candidate_passages = await retrieve_english_passages(retrieval_query, config=config, resources=resources)
    else:
        retrieval_query = retrieval_query_text(row)
        retrieval_scope = "multilingual"
        candidate_passages = await retrieve_multilingual_passages(retrieval_query, config=config, row=row, resources=resources)
    passages = lexical_rerank(
        row,
        candidate_passages,
        final_top_k=final_top_k,
        reranker=reranker,
        query_text=retrieval_query,
    )
    processed = await process_passages(row, method, passages, model=model, config=config, row_key=key)
    kept = [item for item in processed if item.keep]
    max_context_chars_per_doc = int(config.get("max_context_chars_per_doc") or 1200)
    contexts = [
        truncate_context_text(item.context_text(method=method), max_chars=max_context_chars_per_doc)
        for item in kept
    ]
    answer_payload = await model.json(
        solver_messages(
            row,
            method=method,
            retrieval_query=retrieval_query,
            contexts=contexts,
            translated_query=translated_query_payload,
            retrieval_scope=retrieval_scope,
        ),
        namespace=f"{key}.answer",
    )
    pred, valid = parse_answer_label(answer_payload, row)
    gold = str(row.get("answer") or "").strip().upper()
    return {
        "eval_key": key,
        "method": method,
        "dataset": row.get("_dataset") or row.get("dataset") or "global_mmlu",
        "language": row.get("language") or row.get("_language"),
        "subject": row.get("subject"),
        "sample_id": row.get("sample_id") or row.get("question_id") or row.get("id"),
        "answer": gold,
        "prediction": pred,
        "valid": valid,
        "correct": bool(valid and pred == gold),
        "retrieval_query": retrieval_query,
        "retrieval_scope": retrieval_scope,
        "candidate_top_k": candidate_top_k,
        "final_top_k": final_top_k,
        "translated_query": translated_query_payload,
        "candidate_count": len(candidate_passages),
        "retrieved_candidates": [
            {
                "passage_id": passage.passage_id,
                "title": passage.title,
                "section": passage.section,
                "score": passage.score,
                "language": passage.language,
                "rank": passage.rank,
                "text_preview": passage.text[:300],
            }
            for passage in candidate_passages
        ],
        "retrieved": [
            {
                "passage_id": passage.passage_id,
                "title": passage.title,
                "section": passage.section,
                "score": passage.score,
                "language": passage.language,
                "rank": passage.rank,
                "text_preview": passage.text[:300],
            }
            for passage in passages
        ],
        "processed_context": [
            {
                "passage_id": item.passage.passage_id,
                "outcome": item.outcome,
                "keep": item.keep,
                "quality_scores": item.quality_scores,
                "translated_preview": (item.translated_text or "")[:300],
                "refined_preview": (item.refined_text or "")[:300],
            }
            for item in processed
        ],
        "latency_s": time.perf_counter() - started,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    valid = sum(1 for row in rows if row.get("valid"))
    by_method: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("method"))].append(row)
    for method, items in sorted(grouped.items()):
        n = len(items)
        by_method.append(
            {
                "method": method,
                "n": n,
                "correct": sum(1 for item in items if item.get("correct")),
                "accuracy": (sum(1 for item in items if item.get("correct")) / n) if n else 0.0,
                "parse_rate": (sum(1 for item in items if item.get("valid")) / n) if n else 0.0,
            }
        )
    by_language = []
    grouped_lang: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_lang[(str(row.get("method")), str(row.get("language")))].append(row)
    for (method, language), items in sorted(grouped_lang.items()):
        n = len(items)
        by_language.append(
            {
                "method": method,
                "language": language,
                "n": n,
                "correct": sum(1 for item in items if item.get("correct")),
                "accuracy": (sum(1 for item in items if item.get("correct")) / n) if n else 0.0,
            }
        )
    return {
        "n": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parse_rate": valid / total if total else 0.0,
        "by_method": by_method,
        "by_language": by_language,
        "outcome_counts": dict(Counter(item.get("outcome") for row in rows for item in row.get("processed_context", []))),
    }


async def async_main(args: argparse.Namespace) -> int:
    config = merged_config(args)
    output_dir = Path(config.get("output_dir") or "runs/smoke_001/rag_baselines_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(config)
    methods: list[RagMethod] = list(config.get("methods") or ["trag", "dkm_rag", "qtt_rag"])
    final_top_k = int(config.get("final_top_k") or config.get("generation_top_k") or 5)
    candidate_top_k = int(config.get("candidate_top_k") or config.get("top_k") or 50)
    estimated = estimate_model_calls(rows, methods, final_top_k)
    max_model_calls = int(config.get("max_model_calls") or 800)
    requested_concurrency = int(config.get("max_concurrent_model_requests") or 4)
    max_concurrency = requested_concurrency if config.get("allow_full_run") else min(requested_concurrency, 4)
    plan = {
        "rows": len(rows),
        "methods": methods,
        "estimated_model_calls": estimated,
        "max_model_calls": max_model_calls,
        "candidate_top_k": candidate_top_k,
        "final_top_k": final_top_k,
        "live_model": bool(config.get("live_model")),
        "allow_full_run": bool(config.get("allow_full_run")),
        "retrieval_mode": config.get("retrieval_mode"),
        "reranker": config.get("reranker") or "lexical",
        "output_dir": str(output_dir),
    }
    print(json.dumps({"event": "plan", **plan}, ensure_ascii=False), flush=True)
    write_jsonl(output_dir / "plan.jsonl", [plan])
    if estimated["total"] > max_model_calls:
        raise RuntimeError(f"refusing run: estimated model calls {estimated['total']} > max_model_calls {max_model_calls}")
    if config.get("dry_run_plan"):
        return 0

    prediction_path = output_dir / "predictions.jsonl"
    error_path = output_dir / "errors.jsonl"
    existing = {str(row.get("eval_key")): row for row in read_jsonl(prediction_path)} if config.get("resume") and prediction_path.exists() else {}
    if not config.get("resume"):
        if prediction_path.exists():
            prediction_path.unlink()
        if error_path.exists():
            error_path.unlink()
    resources = RetrievalResources(cache=JsonCache(output_dir / "cache" / "retrieval"))
    if config.get("retrieval_mode") != "mock" and any(method in methods for method in ["dkm_rag", "qtt_rag"]):
        resources.multilingual = MultilingualFaissRetriever(
            root=Path(config.get("multilingual_root") or "/tmp/ca_multilingual_wikipedia_qwen3_4b/full_20231101"),
            languages=list(config.get("languages") or ["bn", "hi", "sw", "te", "ne"]),
            model_dir=Path(config.get("embedding_model_dir") or "data/external/models/Qwen3-Embedding-4B"),
            device=str(config.get("embedding_device") or "cuda"),
            cache=resources.cache,
            ef_search=int(config.get("ef_search") or 128),
        )
    if config.get("live_model") and config.get("local_vllm"):
        model: MockModel | LiveModel | LocalVLLMModel = LocalVLLMModel(
            model_dir=Path(config.get("local_model_dir") or "/tmp/qwen_models/Qwen3-8B"),
            model_name=str(config.get("local_model_name") or "Qwen3-8B"),
            max_model_len=int(config.get("local_max_model_len") or 8192),
            gpu_memory_utilization=float(config.get("local_gpu_memory_utilization") or 0.70),
            batch_max_tokens=int(config.get("batch_max_tokens") or 4096),
            local_batch_size=int(config.get("local_batch_size") or 1),
            local_batch_timeout_ms=int(config.get("local_batch_timeout_ms") or 0),
        )
    elif config.get("live_model"):
        model = LiveModel(output_dir=output_dir, concurrency=max_concurrency)
    else:
        model = MockModel()
    results: list[dict[str, Any]] = list(existing.values())
    errors: list[dict[str, Any]] = []
    try:
        new_results: list[dict[str, Any]] = []
        canary_total = 0
        canary_failures = 0
        row_semaphore = asyncio.Semaphore(max_concurrency)

        async def process_row(method: RagMethod, row: dict[str, Any]) -> dict[str, Any]:
            key = eval_key(method, row)
            async with row_semaphore:
                try:
                    return await run_one(row, method, model=model, config=config, resources=resources)
                except Exception as exc:
                    result = {
                        "eval_key": key,
                        "method": method,
                        "language": row.get("language"),
                        "subject": row.get("subject"),
                        "sample_id": row.get("sample_id"),
                        "valid": False,
                        "correct": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    errors.append(result)
                    return result

        with prediction_path.open("a", encoding="utf-8") as prediction_file, error_path.open("a", encoding="utf-8") as error_file:
            for method in methods:
                pending_rows = [row for row in rows if eval_key(method, row) not in existing]
                tasks = [asyncio.create_task(process_row(method, row)) for row in pending_rows]
                for task in asyncio.as_completed(tasks):
                    result = await task
                    key = str(result.get("eval_key"))
                    new_results.append(result)
                    prediction_file.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    prediction_file.flush()
                    if result.get("error"):
                        error_file.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                        error_file.flush()
                    if canary_total < 20:
                        canary_total += 1
                        canary_failures += int(bool(result.get("error")))
                        if canary_total == 20 and canary_failures / canary_total > 0.20:
                            for pending in tasks:
                                if not pending.done():
                                    pending.cancel()
                            raise RuntimeError(f"canary failure rate too high: {canary_failures}/{canary_total}")
                    print(json.dumps({"event": "item_done", "eval_key": key, "correct": result.get("correct"), "error": result.get("error")}, ensure_ascii=False), flush=True)
        results.extend(new_results)
    finally:
        await model.aclose()
        resources.close()

    write_jsonl(prediction_path, results)
    if errors:
        write_jsonl(error_path, errors)
    summary = summarize(results)
    summary["model_calls_observed"] = model.calls
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report_lines = [
        "# Adapted multilingual RAG baseline smoke",
        "",
        f"- rows: {len(rows)}",
        f"- methods: {', '.join(methods)}",
        f"- live_model: {bool(config.get('live_model'))}",
        f"- retrieval_mode: {config.get('retrieval_mode')}",
        f"- model_calls_observed: {model.calls}",
        "",
        "| method | n | accuracy | parse_rate |",
        "|---|---:|---:|---:|",
    ]
    for item in summary["by_method"]:
        report_lines.append(f"| {item['method']} | {item['n']} | {item['accuracy']:.4f} | {item['parse_rate']:.4f} |")
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", "summary": summary}, ensure_ascii=False), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
