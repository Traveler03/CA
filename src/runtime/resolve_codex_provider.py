from __future__ import annotations

import argparse
import json
import os
import subprocess
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.utils.secrets import mask_secret, host_from_url


DEFAULT_CHAT_MODEL = "qwen3.5-9b"
DEFAULT_EMBEDDING_MODEL = "compass-embedding-v4"


@dataclass
class ResolvedProvider:
    base_url: str
    base_url_host: str
    base_url_source: str
    key_source: str
    key_masked: str
    chat_model: str
    chat_model_source: str
    embedding_model: str
    embedding_model_source: str
    model_provider: str | None = None
    wire_api: str | None = None
    checked_locations: list[str] = field(default_factory=list)

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProviderResolutionError(RuntimeError):
    def __init__(self, message: str, checked_locations: list[str]):
        super().__init__(message)
        self.checked_locations = checked_locations


def _project_env_files(start: Path) -> list[Path]:
    return [start / ".env", start / ".env.local"]


def _candidate_config_paths(cwd: Path) -> list[Path]:
    return [
        Path("/home/work/.codex/config.toml"),
        Path.home() / ".codex" / "config.toml",
        Path.home() / ".config" / "codex" / "config.toml",
        *_project_env_files(cwd),
    ]


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def _first_env(names: list[str], env: dict[str, str]) -> tuple[str, str] | None:
    for name in names:
        value = env.get(name)
        if value:
            return name, value
    return None


def _normalise_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        return value
    parsed = urlparse(value)
    if not parsed.scheme:
        value = "https://" + value
    return value.rstrip("/")


def _read_config_auth(auth: dict[str, Any] | None) -> tuple[str, str] | None:
    """Read a key from a Codex auth stanza without exposing it.

    Supports the auth forms seen in Codex configs:
    - {"env_key": "NAME"}
    - {"command": "/bin/cat", "args": ["/path/to/key"]}

    Arbitrary auth commands are intentionally not executed.
    """
    if not auth:
        return None
    env_key = auth.get("env_key") or auth.get("env_var") or auth.get("environment")
    if isinstance(env_key, str) and os.environ.get(env_key):
        return f"environment:{env_key}", os.environ[env_key]

    command = auth.get("command")
    args = auth.get("args") or []
    if command in {"/bin/cat", "cat"} and len(args) == 1:
        key_path = Path(str(args[0])).expanduser()
        if key_path.exists():
            return f"codex_auth_file:{key_path}", key_path.read_text(encoding="utf-8").strip()

    if isinstance(command, str) and args:
        # Some configs use a trusted helper. Avoid shell execution and cap output.
        try:
            completed = subprocess.run(
                [command, *map(str, args)],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        key = completed.stdout.strip()
        if key:
            return f"codex_auth_command:{Path(command).name}", key
    return None


def _provider_key_names(provider_name: str | None) -> list[str]:
    names = [
        "OPENAI_API_KEY",
        "COMPASS_API_KEY",
        "QWEN_API_KEY",
        "DASHSCOPE_API_KEY",
        "ALIYUN_API_KEY",
        "API_KEY",
    ]
    if provider_name:
        normalized = "".join(ch if ch.isalnum() else "_" for ch in provider_name.upper()).strip("_")
        if normalized != "OPENAI":
            names.insert(1, f"{normalized}_API_KEY")
    # Keep order while deduplicating.
    return list(dict.fromkeys(names))


def _base_url_names(provider_name: str | None) -> list[str]:
    names = [
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "BASE_URL",
        "QWEN_BASE_URL",
        "DASHSCOPE_BASE_URL",
        "COMPASS_BASE_URL",
    ]
    if provider_name:
        normalized = "".join(ch if ch.isalnum() else "_" for ch in provider_name.upper()).strip("_")
        names.insert(0, f"{normalized}_BASE_URL")
    return list(dict.fromkeys(names))


def resolve_provider(
    cwd: str | Path = ".",
    *,
    base_url_override: str | None = None,
    api_key_env_override: str | None = None,
    chat_model_override: str | None = None,
    embedding_model_override: str | None = None,
) -> tuple[ResolvedProvider, str]:
    cwd = Path(cwd).resolve()
    env = dict(os.environ)
    checked: list[str] = []

    config: dict[str, Any] = {}
    config_path: Path | None = None
    dotenv_values: dict[str, str] = {}
    for path in _candidate_config_paths(cwd):
        checked.append(str(path))
        if path.name.startswith(".env"):
            dotenv_values.update(_load_dotenv(path))
        elif path.exists() and not config:
            config_path = path
            config = _load_toml(path)

    merged_env = {**dotenv_values, **env}
    provider_name = config.get("model_provider") if isinstance(config.get("model_provider"), str) else None
    providers = config.get("model_providers") if isinstance(config.get("model_providers"), dict) else {}
    provider_config = providers.get(provider_name, {}) if provider_name else {}
    if not isinstance(provider_config, dict):
        provider_config = {}

    if base_url_override:
        base_url = _normalise_base_url(base_url_override)
        base_url_source = "override"
    else:
        env_url = _first_env(_base_url_names(provider_name), merged_env)
        if env_url:
            base_url_source, base_url = f"environment:{env_url[0]}", _normalise_base_url(env_url[1])
        else:
            configured_url = provider_config.get("base_url") or config.get("base_url")
            if isinstance(configured_url, str) and configured_url:
                base_url = _normalise_base_url(configured_url)
                base_url_source = f"codex_config:{config_path}"
            else:
                raise ProviderResolutionError(
                    "Could not resolve API base_url; refusing to fall back to a public default.",
                    checked,
                )

    key_pair: tuple[str, str] | None = None
    if api_key_env_override:
        key = merged_env.get(api_key_env_override)
        if key:
            key_pair = (f"environment:{api_key_env_override}", key)
    if key_pair is None:
        env_key = _first_env(_provider_key_names(provider_name), merged_env)
        if env_key:
            key_pair = (f"environment:{env_key[0]}", env_key[1])
    if key_pair is None:
        auth = provider_config.get("auth") if isinstance(provider_config.get("auth"), dict) else None
        key_pair = _read_config_auth(auth)
    if key_pair is None or not key_pair[1]:
        raise ProviderResolutionError(
            "Could not resolve API key; checked environment, Codex config auth, and project .env files.",
            checked,
        )

    chat_model = (
        chat_model_override
        or merged_env.get("CHAT_MODEL")
        or merged_env.get("QWEN_CHAT_MODEL")
        or DEFAULT_CHAT_MODEL
    )
    chat_model_source = (
        "override"
        if chat_model_override
        else "environment"
        if merged_env.get("CHAT_MODEL") or merged_env.get("QWEN_CHAT_MODEL")
        else "default:qwen3.5-9b"
    )
    embedding_model = (
        embedding_model_override
        or merged_env.get("EMBEDDING_MODEL")
        or merged_env.get("QWEN_EMBEDDING_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    )
    embedding_model_source = (
        "override"
        if embedding_model_override
        else "environment"
        if merged_env.get("EMBEDDING_MODEL") or merged_env.get("QWEN_EMBEDDING_MODEL")
        else f"default:{DEFAULT_EMBEDDING_MODEL}"
    )

    resolved = ResolvedProvider(
        base_url=base_url,
        base_url_host=host_from_url(base_url),
        base_url_source=base_url_source,
        key_source=key_pair[0],
        key_masked=mask_secret(key_pair[1]),
        chat_model=chat_model,
        chat_model_source=chat_model_source,
        embedding_model=embedding_model,
        embedding_model_source=embedding_model_source,
        model_provider=provider_name,
        wire_api=provider_config.get("wire_api") if isinstance(provider_config.get("wire_api"), str) else None,
        checked_locations=checked,
    )
    return resolved, key_pair[1]


def write_resolved_provider(path: str | Path, resolved: ResolvedProvider) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resolved.public_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve the current Codex OpenAI-compatible provider.")
    parser.add_argument("--output", default="artifacts/runtime/resolved_provider.json")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--chat-model")
    parser.add_argument("--embedding-model")
    parser.add_argument("--cwd", default=".")
    args = parser.parse_args(argv)

    try:
        resolved, _key = resolve_provider(
            args.cwd,
            base_url_override=args.base_url,
            api_key_env_override=args.api_key_env,
            chat_model_override=args.chat_model,
            embedding_model_override=args.embedding_model,
        )
    except ProviderResolutionError as exc:
        payload = {"error": str(exc), "checked_locations": exc.checked_locations}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 2

    write_resolved_provider(args.output, resolved)
    print(json.dumps(resolved.public_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
