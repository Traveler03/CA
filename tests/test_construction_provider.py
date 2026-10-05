import json

from src.runtime import resolve_codex_provider as provider
from src.construction.config import ModelConfig
from src.construction.model_client import _extract_content, build_chat_payload


def test_configured_credentials_can_be_reused_without_inheriting_model(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('model = "configured-model"\nmodel_provider = "test"\n'
                    '[model_providers.test]\nbase_url = "https://example.invalid/v1"\n'
                    'wire_api = "responses"\nexperimental_bearer_token = "fixture-secret"\n')
    monkeypatch.setattr(provider, "_candidate_config_paths", lambda _: [path])
    monkeypatch.setattr(provider.os, "environ", {})
    resolved, key = provider.resolve_provider(tmp_path, prefer_config_model=True)
    assert key == "fixture-secret"
    assert resolved.chat_model == "configured-model" and resolved.wire_api == "responses"
    assert "fixture-secret" not in json.dumps(resolved.public_dict())
    default, _ = provider.resolve_provider(tmp_path)
    assert default.chat_model == provider.DEFAULT_CHAT_MODEL == "qwen3.5-9b"
    overridden, _ = provider.resolve_provider(tmp_path, prefer_config_model=True, chat_model_override="override")
    assert overridden.chat_model == "override"


def test_explicit_environment_credentials_and_model_take_precedence(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('model = "configured-model"\nmodel_provider = "test"\n'
                    '[model_providers.test]\nbase_url = "https://example.invalid/v1"\n'
                    'experimental_bearer_token = "config-secret"\n')
    monkeypatch.setattr(provider, "_candidate_config_paths", lambda _: [path])
    monkeypatch.setattr(provider.os, "environ", {"OPENAI_API_KEY": "environment-secret", "CHAT_MODEL": "environment-model"})
    resolved, key = provider.resolve_provider(tmp_path, prefer_config_model=True)
    assert key == "environment-secret" and resolved.chat_model == "environment-model"


def test_responses_payload_and_typed_output_extraction():
    messages = [{"role": "user", "content": "Return JSON."}]
    config = ModelConfig(name="qwen3.5-9b", endpoint="responses", max_completion_tokens=4096, reasoning_effort=None)
    payload = build_chat_payload(config, messages)
    assert payload["input"] == messages and payload["max_output_tokens"] == 4096
    assert "reasoning" not in payload and payload["store"] is False
    assert "messages" not in payload and "max_completion_tokens" not in payload
    raw = {"output": [{"type": "reasoning", "summary": []},
                      {"type": "message", "content": [{"type": "output_text", "text": '{"ok":true}'}]}]}
    assert json.loads(_extract_content(raw)) == {"ok": True}
    chat = build_chat_payload(ModelConfig(), messages)
    assert chat["model"] == "qwen3.5-9b" and chat["messages"] == messages and "max_tokens" in chat
    assert chat["reasoning_effort"] == "none" and "max_completion_tokens" not in chat
    assert _extract_content({"choices": [{"message": {"content": "final response"}}]}) == "final response"


def test_cli_defaults_do_not_inherit_provider_model_or_protocol(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import build_wikipag_application_cards as cli
    from test_wikipag_application_cards import group

    captured = {}
    def resolve(cwd, **kwargs):
        captured["provider_request"] = kwargs
        return SimpleNamespace(chat_model=kwargs["chat_model_override"],
                               base_url="https://example.invalid/v1", wire_api="responses"), "fixture-secret"
    class Client:
        def __init__(self, **kwargs):
            captured["config"] = kwargs["model_config"]
        async def aclose(self):
            captured["closed"] = True
    class Builder:
        def __init__(self, *args, **kwargs):
            pass
        async def run(self, groups):
            return {"complete": True}
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "resolve_provider", resolve)
    monkeypatch.setattr(cli, "ApplicationModelClient", Client)
    monkeypatch.setattr(cli, "ApplicationCardBuilder", Builder)
    path = tmp_path / "inputs.jsonl"
    path.write_text(json.dumps(group().model_dump()) + "\n")
    assert cli.main(["--passage-groups", str(path), "--output-dir", str(tmp_path / "runs/new")]) == 0
    assert captured["provider_request"]["chat_model_override"] == "qwen3.5-9b"
    assert captured["config"].name == "qwen3.5-9b" and captured["config"].endpoint == "chat/completions"
    assert captured["config"].reasoning_effort == "none" and captured["closed"]
