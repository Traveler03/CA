from __future__ import annotations

import argparse
import concurrent.futures as cf
import subprocess
import sys
import time
from pathlib import Path

import requests


FAISS_REPO = "Sherlock-Comms/wikipedia-en-2026-07-01-faiss"
PASSAGES_REPO = "Sherlock-Comms/wikipedia-en-2026-07-01-passages"
MODEL_REPO = "Qwen/Qwen3-Embedding-4B"


def fetch_siblings(repo: str, *, kind: str, insecure: bool) -> list[dict]:
    url = f"https://huggingface.co/api/{kind}/{repo}?blobs=true"
    response = requests.get(url, timeout=60, verify=not insecure)
    response.raise_for_status()
    return list(response.json()["siblings"])


def fmt_bytes(value: int) -> str:
    n = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.2f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1000
    return f"{n:.2f}TB"


def resolve_tasks(args: argparse.Namespace) -> list[tuple[str, str, Path, int, str]]:
    root = Path(args.output_root)
    tasks: list[tuple[str, str, Path, int, str]] = []

    faiss_dir = root / "wikipedia-en-2026-07-01-faiss"
    faiss_sizes = {s["rfilename"]: int(s.get("size") or 0) for s in fetch_siblings(FAISS_REPO, kind="datasets", insecure=args.insecure)}
    faiss_names = ["README.md", "ivfpq.faiss", "ids.txt", "offsets.sqlite"]
    if args.hnsw:
        faiss_names.append("hnsw_sq.faiss")
    for name in faiss_names:
        tasks.append((FAISS_REPO, name, faiss_dir / name, faiss_sizes[name], "datasets"))

    if args.passages:
        passages_dir = root / "wikipedia-en-2026-07-01-passages"
        for s in fetch_siblings(PASSAGES_REPO, kind="datasets", insecure=args.insecure):
            name = s["rfilename"]
            if name == "README.md" or name.endswith(".jsonl"):
                tasks.append((PASSAGES_REPO, name, passages_dir / name, int(s.get("size") or 0), "datasets"))

    if args.model:
        model_dir = root / "models" / "Qwen3-Embedding-4B"
        for s in fetch_siblings(MODEL_REPO, kind="models", insecure=args.insecure):
            name = s["rfilename"]
            tasks.append((MODEL_REPO, name, model_dir / name, int(s.get("size") or 0), "models"))

    return tasks


def download_one(task: tuple[str, str, Path, int, str], *, log_dir: Path) -> tuple[str, str, int, str]:
    repo, name, dest, expected, kind = task
    if dest.exists() and expected and dest.stat().st_size == expected:
        return ("skip", str(dest), expected, "")

    dest.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (str(dest).replace("/", "_") + ".curl.log")
    url = f"https://huggingface.co/{'datasets/' if kind == 'datasets' else ''}{repo}/resolve/main/{name}"
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "10",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "-C",
        "-",
        "-o",
        str(dest),
        url,
    ]
    started = time.time()
    with log_path.open("ab") as log:
        completed = subprocess.run(cmd, stdout=log, stderr=log)
    got = dest.stat().st_size if dest.exists() else 0
    if completed.returncode != 0:
        return ("fail", str(dest), got, f"rc={completed.returncode} log={log_path}")
    if expected and got != expected:
        return ("size_mismatch", str(dest), got, f"expected={expected} log={log_path}")
    return ("done", str(dest), got, f"{time.time() - started:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Sherlock-Comms English Wikipedia FAISS RAG assets.")
    parser.add_argument("--output-root", default="data/external")
    parser.add_argument("--log-dir", default="artifacts/wiki_faiss_20260701/download_logs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification for Hugging Face metadata API.")
    parser.add_argument("--no-passages", dest="passages", action="store_false", default=True)
    parser.add_argument("--no-model", dest="model", action="store_false", default=True)
    parser.add_argument("--hnsw", action="store_true", help="Also download the 40GB hnsw_sq.faiss index.")
    args = parser.parse_args()

    tasks = resolve_tasks(args)
    total = sum(size for *_rest, size, _kind in tasks)
    print(f"tasks={len(tasks)} expected={fmt_bytes(total)} workers={args.workers}")
    for repo, name, dest, size, kind in tasks:
        print(f"task {kind}:{repo}/{name} {fmt_bytes(size)} -> {dest}")
    sys.stdout.flush()

    failures = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, task, log_dir=Path(args.log_dir)) for task in tasks]
        for future in cf.as_completed(futures):
            status, path, got, detail = future.result()
            print(status, fmt_bytes(got), path, detail)
            sys.stdout.flush()
            if status in {"fail", "size_mismatch"}:
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
