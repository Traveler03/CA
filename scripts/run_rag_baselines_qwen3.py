#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.methods.rag_baselines.pipeline import load_config, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen3 multilingual RAG baselines: TRAG, DKM-RAG, QTT-RAG.")
    parser.add_argument("--config", type=Path, default=Path("configs/rag_baselines_qwen3_8b.yaml"))
    parser.add_argument("--preset", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--run-name")
    parser.add_argument("--methods", help="Comma-separated subset: trag,dkm_rag,qtt_rag")
    parser.add_argument("--limit-per-language", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--model-dir")
    parser.add_argument("--global-mmlu-dir")
    parser.add_argument("--mmlu-prox-dir")
    parser.add_argument("--embedding-model-dir")
    parser.add_argument("--multilingual-root")
    parser.add_argument("--english-service-url")
    parser.add_argument("--retrieval-device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config: dict[str, Any] = load_config(args.config, preset=args.preset)
    if args.run_name:
        config["run_name"] = args.run_name
    if args.methods:
        config["methods"] = [item.strip() for item in args.methods.split(",") if item.strip()]
    if args.limit_per_language is not None:
        config["limit_per_language"] = args.limit_per_language
    if args.output_root:
        config["output_root"] = args.output_root
    if args.model_dir:
        config.setdefault("llm", {})["model_dir"] = args.model_dir
    if args.global_mmlu_dir:
        config.setdefault("data", {})["global_mmlu_dir"] = args.global_mmlu_dir
    if args.mmlu_prox_dir:
        config.setdefault("data", {})["mmlu_prox_dir"] = args.mmlu_prox_dir
    if args.embedding_model_dir:
        config.setdefault("retrieval", {})["embedding_model_dir"] = args.embedding_model_dir
    if args.multilingual_root:
        config.setdefault("retrieval", {})["multilingual_root"] = args.multilingual_root
    if args.english_service_url:
        config.setdefault("retrieval", {})["english_service_url"] = args.english_service_url
    if args.retrieval_device:
        config.setdefault("retrieval", {})["device"] = args.retrieval_device
    if args.batch_size is not None:
        config.setdefault("llm", {})["batch_size"] = args.batch_size
    if args.gpu_memory_utilization is not None:
        config.setdefault("llm", {})["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.no_resume:
        config["resume"] = False

    # Keep Qwen3 thinking disabled unless a caller explicitly overrides in the environment.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    manifest = run_pipeline(args.config, config)
    print(json.dumps({"status": "ok", "output_dir": manifest["output_dir"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
