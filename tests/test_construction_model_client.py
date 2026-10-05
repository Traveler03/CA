import asyncio
import json

import httpx
import pytest

from src.construction.config import ModelConfig
from src.construction.model_client import ApplicationModelClient, parse_json_object


def client_with_responses(tmp_path, monkeypatch, responses):
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        status, response = responses[min(len(requests) - 1, len(responses) - 1)]
        return httpx.Response(status, json=response)
    factory = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: factory(
        **kwargs, transport=httpx.MockTransport(handle)))
    client = ApplicationModelClient(model_config=ModelConfig(max_retries=0),
                                    base_url="https://example.invalid/v1", api_key="fixture-secret",
                                    cache_dir=tmp_path / "cache", raw_dir=tmp_path / "raw")
    return client, requests


def chat(content='{"ok":true}', finish_reason="stop"):
    return {"model": "qwen3.5-9b", "choices": [{"finish_reason": finish_reason,
            "message": {"reasoning_content": '{"ok":false}', "content": content}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}


def test_qwen_final_json_and_successful_response_cache(tmp_path, monkeypatch):
    raw = chat('<think>Consider {"ok":false} but reject it.</think>\n{"ok":true}')
    client, requests = client_with_responses(tmp_path, monkeypatch, [(200, raw)])
    messages = [{"role": "user", "content": "Return JSON."}]
    async def run():
        first, result, error = await client.complete_json(messages, namespace="test", validate=lambda x: x)
        second, cached, _ = await client.complete_json(messages, namespace="test", validate=lambda x: x)
        assert first == second == {"ok": True} and error is None
        assert not result.cached and cached.cached
    asyncio.run(run())
    assert len(requests) == 1 and requests[0]["model"] == "qwen3.5-9b"
    assert requests[0]["max_tokens"] == 6144 and requests[0]["reasoning_effort"] == "none"
    assert client.usage_summary()["total_tokens"] == 18
    assert "fixture-secret" not in "".join(p.read_text() for p in tmp_path.rglob("*.json"))


@pytest.mark.parametrize("failure", [
    (200, chat(finish_reason="length")),
    (200, chat(finish_reason="content_filter")),
    (200, {"status": "incomplete", "output_text": '{"ok":true}'}),
    (503, {"error": {"message": "Temporary unavailable"}}),
])
def test_incomplete_or_failed_calls_are_not_accepted_or_cached(tmp_path, monkeypatch, failure):
    client, requests = client_with_responses(tmp_path, monkeypatch, [failure, (200, chat())])
    async def run():
        first = await client.complete_once([{"role": "user", "content": "Return JSON."}], namespace="retry")
        assert not first.ok and not client.cache.path_for(first.cache_key).exists()
        assert list((tmp_path / "raw").glob("*.json"))
        second = await client.complete_once([{"role": "user", "content": "Return JSON."}], namespace="retry")
        assert second.ok and not second.cached
        assert len(list((tmp_path / "raw").glob("*.json"))) == 2
    asyncio.run(run())
    assert len(requests) == 2


def test_unfinished_thinking_cannot_be_parsed_as_the_answer():
    with pytest.raises(ValueError, match="unfinished thinking"):
        parse_json_object('<think>Candidate answer: {"ok":true}')


def test_format_retry_includes_previous_json_and_specific_error(tmp_path, monkeypatch):
    client, requests = client_with_responses(tmp_path, monkeypatch, [
        (200, chat('{"name":"stable"}')), (200, chat('{"name":"stable","definition":"present"}'))])
    client.model_config.max_retries = 1
    async def no_delay(_):
        pass
    monkeypatch.setattr(asyncio, "sleep", no_delay)
    def validate(payload):
        if "definition" not in payload:
            raise ValueError("missing definition[null]")
        return payload
    payload, _, error = asyncio.run(client.complete_json(
        [{"role": "user", "content": "Return both fields."}], namespace="schema", validate=validate))
    assert payload == {"name": "stable", "definition": "present"} and error is None
    correction = requests[1]["messages"]
    assert correction[-2] == {"role": "assistant", "content": '{"name":"stable"}'}
    assert "missing definition[null]" in correction[-1]["content"]
