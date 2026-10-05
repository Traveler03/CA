"""Build five-field English Wikipag cards with Qwen3.5-9B application synthesis."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.construction.application_cards import ApplicationCardBuilder, load_groups
from src.construction.model_client import ApplicationModelClient
from src.construction.source_preparation import SourcePreparer
from src.runtime.resolve_codex_provider import resolve_provider
from src.construction.config import DEFAULT_MODEL, ModelConfig


async def build(args: argparse.Namespace) -> dict:
    groups = load_groups(args.passage_groups)
    if args.validate_only:
        return {"validation": "passed", "groups": len(groups), "source_corpus": "English Wikipag",
                "model_calls": 0}
    output = args.output_dir.resolve()
    runs = (ROOT / "runs").resolve()
    if output == runs or not output.is_relative_to(runs):
        raise ValueError("--output-dir must be a dedicated directory under this repository's runs/")
    if output.exists() and not (output / "construction_manifest.json").exists() and any(output.iterdir()):
        raise ValueError("output directory already contains data; choose an empty construction run")
    preparer = SourcePreparer(offsets_db=args.offsets_db, corpus_root=args.corpus_root,
                              source_id_prefix=args.source_id_prefix,
                              max_candidates=args.max_source_candidates, max_passages=args.max_passages)
    # Read endpoint credentials from the provider without inheriting its model/protocol.
    resolved, key = resolve_provider(str(ROOT), chat_model_override=args.model,
                                     base_url_override=args.base_url, api_key_env_override=args.api_key_env)
    config = ModelConfig(name=resolved.chat_model, concurrency=args.concurrency,
                         endpoint=args.endpoint,
                         max_completion_tokens=args.max_completion_tokens,
                         reasoning_effort=args.reasoning_effort, max_retries=args.max_retries,
                         cache_enabled=True)
    client = ApplicationModelClient(
        model_config=config, base_url=resolved.base_url, api_key=key,
        cache_dir=output / "cache" / "model", raw_dir=output / "model_raw", timeout_s=args.timeout,
    )
    try:
        builder = ApplicationCardBuilder(client, output,
                                        provider_id=hashlib.sha256(resolved.base_url.encode()).hexdigest(),
                                        preparer=preparer, max_card_repairs=args.max_card_repairs)
        return await builder.run(groups)
    finally:
        await client.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passage-groups", type=Path, required=True,
                        help="JSONL of topic-related English Wikipag passage groups; see docs/CONSTRUCTION.md")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "wikipag_qwen35_cards")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Qwen3.5-9B service model ID shared by every stage (default: qwen3.5-9b)")
    parser.add_argument("--base-url", help="Qwen API base URL; otherwise use configured provider credentials")
    parser.add_argument("--api-key-env", help="Environment variable containing the service API key")
    parser.add_argument("--endpoint", choices=("chat/completions", "responses"), default="chat/completions",
                        help="Explicit Qwen service protocol (default: chat/completions)")
    parser.add_argument("--concurrency", type=int, choices=range(1, 33), default=8)
    parser.add_argument("--max-completion-tokens", type=int, default=6144)
    parser.add_argument("--max-retries", type=int, choices=range(3), default=2)
    parser.add_argument("--max-card-repairs", type=int, choices=(0, 1), default=1,
                        help="At most one repair and re-review after card rejection (default: 1)")
    parser.add_argument("--offsets-db", type=Path, help="Read-only local Wikipag offsets.sqlite for same-article retrieval")
    parser.add_argument("--corpus-root", type=Path, help="Directory containing the original English passage shards")
    parser.add_argument("--source-id-prefix", help="Original corpus source ID prefix, e.g. wikipag-en-2026-07-01:")
    parser.add_argument("--max-source-candidates", type=int, choices=range(1, 257), default=64)
    parser.add_argument("--max-passages", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--reasoning-effort", default="none",
                        help="Service reasoning setting (default: none); solution explanations are still generated")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate all input records without resolving credentials or making model calls")
    args = parser.parse_args(argv)
    if args.max_completion_tokens <= 0 or args.timeout <= 0:
        parser.error("token budget and timeout must be positive")
    if any(x is not None for x in (args.offsets_db, args.corpus_root, args.source_id_prefix)) and not all(
            x is not None for x in (args.offsets_db, args.corpus_root, args.source_id_prefix)):
        parser.error("--offsets-db, --corpus-root and --source-id-prefix must be supplied together")
    result = asyncio.run(build(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("complete", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
