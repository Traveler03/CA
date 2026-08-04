from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


PART_RE = re.compile(r"^(\d+)-(\d+)\.part$")


def fmt_bytes(value: int) -> str:
    n = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.2f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1000
    return f"{n:.2f}TB"


def head_info(url: str, *, insecure: bool) -> tuple[int, bool]:
    response = requests.head(url, allow_redirects=True, timeout=60, verify=not insecure)
    response.raise_for_status()
    size = int(response.headers["content-length"])
    range_ok = "bytes" in (response.headers.get("accept-ranges") or "").lower()
    return size, range_ok


def existing_parts(parts_dir: Path) -> list[tuple[int, int, Path]]:
    parts: list[tuple[int, int, Path]] = []
    if not parts_dir.exists():
        return parts
    for path in parts_dir.iterdir():
        match = PART_RE.match(path.name)
        if not match:
            continue
        start, end = int(match.group(1)), int(match.group(2))
        expected = end - start + 1
        if path.stat().st_size == expected:
            parts.append((start, end, path))
    return sorted(parts)


def normalise_existing_output(output: Path, parts_dir: Path, total_size: int) -> None:
    if not output.exists() or output.stat().st_size == 0:
        return
    if output.stat().st_size == total_size:
        return
    parts_dir.mkdir(parents=True, exist_ok=True)
    size = output.stat().st_size
    part = parts_dir / f"0-{size - 1}.part"
    if not part.exists():
        output.replace(part)
    else:
        output.unlink()


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def missing_ranges(total_size: int, present: list[tuple[int, int]], chunk_size: int) -> list[tuple[int, int]]:
    present = merge_intervals(present)
    missing: list[tuple[int, int]] = []
    cursor = 0
    for start, end in present:
        if cursor < start:
            missing.extend(split_range(cursor, start - 1, chunk_size))
        cursor = max(cursor, end + 1)
    if cursor < total_size:
        missing.extend(split_range(cursor, total_size - 1, chunk_size))
    return missing


def split_range(start: int, end: int, chunk_size: int) -> list[tuple[int, int]]:
    ranges = []
    cursor = start
    while cursor <= end:
        stop = min(end, cursor + chunk_size - 1)
        ranges.append((cursor, stop))
        cursor = stop + 1
    return ranges


def download_range(
    url: str,
    parts_dir: Path,
    start: int,
    end: int,
    *,
    insecure: bool,
    retries: int,
) -> tuple[str, int, int, str]:
    final = parts_dir / f"{start}-{end}.part"
    if final.exists() and final.stat().st_size == end - start + 1:
        return ("skip", start, end, "")
    tmp = parts_dir / f"{start}-{end}.tmp"
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120, verify=not insecure) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"expected 206, got {response.status_code}")
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if tmp.stat().st_size != end - start + 1:
                raise RuntimeError(f"size mismatch {tmp.stat().st_size} != {end - start + 1}")
            tmp.replace(final)
            return ("done", start, end, "")
        except Exception as exc:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            if attempt == retries:
                return ("fail", start, end, str(exc))
            time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    return ("fail", start, end, "unknown")


def assemble(output: Path, parts: list[tuple[int, int, Path]], total_size: int) -> None:
    parts = sorted(parts)
    cursor = 0
    for start, end, _path in parts:
        if start != cursor:
            raise RuntimeError(f"gap before {start}, cursor={cursor}")
        cursor = end + 1
    if cursor != total_size:
        raise RuntimeError(f"incomplete final size cursor={cursor}, total={total_size}")

    tmp_output = output.with_suffix(output.suffix + ".assembling")
    with tmp_output.open("wb") as out:
        for _start, _end, path in parts:
            with path.open("rb") as src:
                shutil.copyfileobj(src, out, length=32 * 1024 * 1024)
    if tmp_output.stat().st_size != total_size:
        raise RuntimeError(f"assembled size mismatch {tmp_output.stat().st_size} != {total_size}")
    tmp_output.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download one large HTTP file with parallel Range requests.")
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=512)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args()

    total_size, range_ok = head_info(args.url, insecure=args.insecure)
    if not range_ok:
        raise RuntimeError("server does not advertise Accept-Ranges: bytes")
    if args.output.exists() and args.output.stat().st_size == total_size:
        print(f"already complete {args.output} {fmt_bytes(total_size)}")
        return 0

    parts_dir = args.output.with_name(args.output.name + ".parts")
    normalise_existing_output(args.output, parts_dir, total_size)
    parts_dir.mkdir(parents=True, exist_ok=True)

    present = [(start, end) for start, end, _path in existing_parts(parts_dir)]
    missing = missing_ranges(total_size, present, args.chunk_mib * 1024 * 1024)
    print(
        f"total={fmt_bytes(total_size)} present={fmt_bytes(sum(end - start + 1 for start, end in present))} "
        f"missing_ranges={len(missing)} workers={args.workers}"
    )
    sys.stdout.flush()

    started = time.perf_counter()
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_range,
                args.url,
                parts_dir,
                start,
                end,
                insecure=args.insecure,
                retries=args.retries,
            )
            for start, end in missing
        ]
        for future in as_completed(futures):
            status, start, end, detail = future.result()
            done_parts = existing_parts(parts_dir)
            done_bytes = sum(e - s + 1 for s, e, _p in done_parts)
            print(
                f"{status} {fmt_bytes(end - start + 1)} bytes={start}-{end} "
                f"progress={fmt_bytes(done_bytes)}/{fmt_bytes(total_size)} elapsed_s={time.perf_counter() - started:.1f} {detail}"
            )
            sys.stdout.flush()
            if status == "fail":
                failures += 1

    if failures:
        return 1

    parts = existing_parts(parts_dir)
    assemble(args.output, parts, total_size)
    if not args.keep_parts:
        shutil.rmtree(parts_dir)
    print(f"assembled {args.output} {fmt_bytes(args.output.stat().st_size)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
