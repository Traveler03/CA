from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from src.clients.cache import JsonRequestCache
from src.runtime.resolve_codex_provider import resolve_provider


@dataclass
class EmbeddingResult:
    ok: bool
    embeddings: list[list[float]]
    raw: dict[str, Any]
    status_code: int | None
    latency_s: float
    attempts: int
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    cache_key: str | None = None


class AsyncEmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        concurrency: int = 4,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        cache_dir: str | Path = "artifacts/cache/embedding",
        raw_dir: str | Path = "artifacts/raw/embedding",
        cache_enabled: bool = True,
        normalize: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.normalize = normalize
        self.semaphore = asyncio.Semaphore(concurrency)
        self.cache = JsonRequestCache(cache_dir, enabled=cache_enabled)
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    @classmethod
    def from_codex_config(
        cls,
        *,
        concurrency: int = 4,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        cache_enabled: bool = True,
    ) -> "AsyncEmbeddingClient":
        resolved, key = resolve_provider(".")
        return cls(
            base_url=resolved.base_url,
            api_key=key,
            model=resolved.embedding_model,
            concurrency=concurrency,
            timeout_s=timeout_s,
            max_retries=max_retries,
            cache_enabled=cache_enabled,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncEmbeddingClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        payload = {"model": self.model, "input": texts}
        cache_key = self.cache.key_for("embedding", payload)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return EmbeddingResult(
                ok=cached.get("ok", False),
                embeddings=cached.get("embeddings", []),
                raw=cached.get("raw", {}),
                status_code=cached.get("status_code"),
                latency_s=0.0,
                attempts=0,
                usage=cached.get("usage", {}),
                error=cached.get("error"),
                cache_key=cache_key,
            )

        result: EmbeddingResult | None = None
        for attempt in range(1, self.max_retries + 2):
            result = await self._send_once(payload, attempt)
            if result.ok:
                break
            if result.status_code is not None and result.status_code not in {429, 500, 502, 503, 504}:
                break
            await asyncio.sleep(_backoff_s(attempt))
        assert result is not None
        result.cache_key = cache_key
        if not result.ok:
            self._save_raw(cache_key, result.raw)
        self.cache.set(cache_key, result.__dict__)
        return result

    async def _send_once(self, payload: dict[str, Any], attempt: int) -> EmbeddingResult:
        async with self.semaphore:
            start = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    self._client.post(f"{self.base_url}/embeddings", json=payload),
                    timeout=self.timeout_s + 5.0,
                )
                latency = time.perf_counter() - start
                try:
                    raw = response.json()
                except Exception:
                    raw = {"text": response.text}
                if response.status_code >= 400:
                    return EmbeddingResult(False, [], raw, response.status_code, latency, attempt, error=_safe_error(raw))
                embeddings = _extract_embeddings(raw)
                if self.normalize and embeddings:
                    embeddings = normalize_vectors(embeddings)
                return EmbeddingResult(
                    ok=bool(embeddings) and len(embeddings) == len(payload["input"]),
                    embeddings=embeddings,
                    raw=raw,
                    status_code=response.status_code,
                    latency_s=latency,
                    attempts=attempt,
                    usage=raw.get("usage", {}) if isinstance(raw.get("usage"), dict) else {},
                    error=None if embeddings else "empty_embeddings",
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                latency = time.perf_counter() - start
                return EmbeddingResult(
                    ok=False,
                    embeddings=[],
                    raw={"exception": type(exc).__name__, "message": str(exc)},
                    status_code=None,
                    latency_s=latency,
                    attempts=attempt,
                    error=type(exc).__name__,
                )

    def _save_raw(self, key: str, raw: dict[str, Any]) -> None:
        path = self.raw_dir / f"{key}.json"
        try:
            path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError:
            return


def _extract_embeddings(raw: dict[str, Any]) -> list[list[float]]:
    data = raw.get("data")
    if not isinstance(data, list):
        return []
    rows: list[tuple[int, list[float]]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        emb = item.get("embedding")
        if isinstance(emb, list) and all(isinstance(x, (int, float)) for x in emb):
            rows.append((int(item.get("index", idx)), [float(x) for x in emb]))
    rows.sort(key=lambda pair: pair[0])
    return [row for _idx, row in rows]


def normalize_vectors(vectors: list[list[float]]) -> list[list[float]]:
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(float).tolist()


def _safe_error(raw: dict[str, Any]) -> str:
    err = raw.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err.get("type") or "api_error")
    return str(err or raw.get("message") or "api_error")


def _backoff_s(attempt: int) -> float:
    return min(30.0, (2 ** (attempt - 1)) + random.uniform(0.0, 0.5))
