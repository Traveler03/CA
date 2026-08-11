from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import httpx

from src.clients.cache import JsonRequestCache
from src.runtime.resolve_codex_provider import resolve_provider


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass
class ChatResult:
    ok: bool
    content: str
    raw: dict[str, Any]
    status_code: int | None
    latency_s: float
    attempts: int
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    invalid_reason: str | None = None
    cache_key: str | None = None


def has_hidden_thinking_marker(text: str) -> bool:
    lowered = text.lower()
    return THINK_OPEN in lowered or THINK_CLOSE in lowered


def has_hidden_reasoning_field(raw: dict[str, Any]) -> bool:
    choices = raw.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        if message.get("reasoning") or message.get("reasoning_content") or message.get("reasoning_details"):
            return True
    return False


def default_chat_payload(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.0,
    response_format: dict[str, Any] | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        # Compass/Qwen currently ignores `chat_template_kwargs` for this model
        # but honors these OpenAI-compatible reasoning controls.
        "reasoning": {"enabled": False},
        "reasoning_effort": "none",
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if logprobs is not None:
        payload["logprobs"] = logprobs
    if top_logprobs is not None:
        payload["top_logprobs"] = top_logprobs
    if extra_body:
        payload.update(dict(extra_body))
    return payload


class AsyncChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        concurrency: int = 4,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        cache_dir: str | Path = "artifacts/cache/chat",
        raw_dir: str | Path = "artifacts/raw/chat",
        cache_enabled: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
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
    ) -> "AsyncChatClient":
        resolved, key = resolve_provider(".")
        return cls(
            base_url=resolved.base_url,
            api_key=key,
            model=resolved.chat_model,
            concurrency=concurrency,
            timeout_s=timeout_s,
            max_retries=max_retries,
            cache_enabled=cache_enabled,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncChatClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def create(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
        json_schema_required_keys: list[str] | None = None,
        retry_on_think: bool = True,
    ) -> ChatResult:
        payload = default_chat_payload(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            extra_body=extra_body,
        )
        cache_key = self.cache.key_for("chat", payload)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return ChatResult(
                ok=cached.get("ok", False),
                content=cached.get("content", ""),
                raw=cached.get("raw", {}),
                status_code=cached.get("status_code"),
                latency_s=0.0,
                attempts=0,
                usage=cached.get("usage", {}),
                error=cached.get("error"),
                invalid_reason=cached.get("invalid_reason"),
                cache_key=cache_key,
            )

        result: ChatResult | None = None
        attempts_allowed = self.max_retries + 1
        think_retry_used = False
        for attempt in range(1, attempts_allowed + 1):
            result = await self._send_once(payload, attempt, cache_key)
            if result.ok:
                if has_hidden_thinking_marker(result.content) or has_hidden_reasoning_field(result.raw):
                    self._save_raw(cache_key + f"-think-attempt-{attempt}", result.raw)
                    if retry_on_think and not think_retry_used:
                        think_retry_used = True
                        continue
                    result.ok = False
                    result.invalid_reason = "hidden_thinking"
                elif json_schema_required_keys is not None:
                    schema_error = _validate_json_keys(result.content, json_schema_required_keys)
                    if schema_error:
                        result.ok = False
                        result.invalid_reason = schema_error
                if result.ok:
                    break
            if (
                result.status_code is not None
                and result.status_code not in {429, 500, 502, 503, 504}
                and result.invalid_reason != "hidden_thinking_marker"
            ):
                break
            await asyncio.sleep(_backoff_s(attempt))

        assert result is not None
        result.cache_key = cache_key
        self._save_raw(cache_key, result.raw)
        self.cache.set(cache_key, result.__dict__)
        return result

    async def _send_once(self, payload: dict[str, Any], attempt: int, cache_key: str) -> ChatResult:
        async with self.semaphore:
            start = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    self._client.post(f"{self.base_url}/chat/completions", json=payload),
                    timeout=self.timeout_s + 5.0,
                )
                latency = time.perf_counter() - start
                raw: dict[str, Any]
                try:
                    raw = response.json()
                except Exception:
                    raw = {"text": response.text}
                if response.status_code >= 400:
                    return ChatResult(
                        ok=False,
                        content="",
                        raw=raw,
                        status_code=response.status_code,
                        latency_s=latency,
                        attempts=attempt,
                        error=_safe_error(raw),
                    )
                content = _extract_content(raw)
                return ChatResult(
                    ok=bool(content),
                    content=content,
                    raw=raw,
                    status_code=response.status_code,
                    latency_s=latency,
                    attempts=attempt,
                    usage=raw.get("usage", {}) if isinstance(raw.get("usage"), dict) else {},
                    error=None if content else "empty_content",
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                latency = time.perf_counter() - start
                return ChatResult(
                    ok=False,
                    content="",
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


def _extract_content(raw: dict[str, Any]) -> str:
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = first.get("text")
            if isinstance(text, str):
                return text
    output = raw.get("output_text")
    if isinstance(output, str):
        return output
    return ""


def _safe_error(raw: dict[str, Any]) -> str:
    err = raw.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err.get("type") or "api_error")
    return str(err or raw.get("message") or "api_error")


def _validate_json_keys(content: str, required_keys: list[str]) -> str | None:
    try:
        parsed = json.loads(content)
    except Exception:
        return "invalid_json"
    if not isinstance(parsed, dict):
        return "json_not_object"
    missing = [key for key in required_keys if key not in parsed]
    if missing:
        return f"missing_json_keys:{','.join(missing)}"
    return None


def _backoff_s(attempt: int) -> float:
    return min(30.0, (2 ** (attempt - 1)) + random.uniform(0.0, 0.5))
