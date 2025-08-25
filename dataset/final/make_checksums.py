#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate and verify checksums + manifest for the public dataset.

Outputs (written next to BASE):
- CHECKSUMS.md      # human-readable table (path, size, sha256[, md5])
- manifest.jsonl    # JSON lines with metadata per file

Usage:
  python3 make_checksums.py                             # generate
  python3 make_checksums.py --verify                    # verify against existing CHECKSUMS.md
  python3 make_checksums.py --base /path/to/final       # custom base dir
  python3 make_checksums.py --include-md5               # also compute md5

Default BASE:
  /mnt/d/new_Fraud/dataset/final
"""

from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

DEFAULT_BASE = Path("/mnt/d/new_Fraud/dataset/final")
CHECKSUMS_NAME = "CHECKSUMS.md"
MANIFEST_NAME = "manifest.jsonl"

# skip patterns (temporary parts, hidden, etc.)
SKIP_DIR_NAMES = {".git", "__pycache__", "_tmp_week_parts", "_tmp_month_parts"}
SKIP_FILE_SUFFIXES = {".tmp", ".part"}


def human_bytes(n: int) -> str:
    # pretty size
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb", buffering=1024*1024) as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb", buffering=1024*1024) as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(base: Path):
    for root, dirs, files in os.walk(base):
        # prune skip dirs
        dirs[:] = [
            d for d in dirs if d not in SKIP_DIR_NAMES and not d.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            p = Path(root) / name
            if any(str(p).endswith(sfx) for sfx in SKIP_FILE_SUFFIXES):
                continue
            yield p


def load_checksums_table(md_path: Path):
    """Parse existing CHECKSUMS.md into dict[path -> (sha256, md5_or_none, size_bytes)]."""
    if not md_path.exists():
        return {}
    out = {}
    with md_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # rows look like: | path | size | sha256 | md5? |
            if not line.startswith("| ") or line.startswith("| Path"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            # expected 4 or 5 columns depending on md5 presence
            if len(cells) < 3:
                continue
            path_cell = cells[0]
            size_cell = cells[1]
            sha_cell = cells[2]
            md5_cell = cells[3] if len(cells) >= 4 else ""
            # strip code fences/backticks if any
            path_cell = path_cell.replace("`", "")
            sha_cell = sha_cell.replace("`", "")
            md5_cell = md5_cell.replace("`", "")
            # size is human-readable; we can ignore exact match on size in verify
            out[path_cell] = (sha_cell, md5_cell or None, size_cell)
    return out


def write_checksums_md(base: Path, rows: list[dict], include_md5: bool):
    md = []
    md.append("# Checksums\n")
    md.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
    md.append("\n")
    if include_md5:
        md.append("| Path | Size | SHA256 | MD5 |\n")
        md.append("|------|------|--------|-----|\n")
    else:
        md.append("| Path | Size | SHA256 |\n")
        md.append("|------|------|--------|\n")
    for r in rows:
        path_rel = r["path_rel"]
        size_hr = r["size_hr"]
        sha256 = r["sha256"]
        if include_md5:
            md5 = r.get("md5", "")
            md.append(f"| `{path_rel}` | {size_hr} | `{sha256}` | `{md5}` |\n")
        else:
            md.append(f"| `{path_rel}` | {size_hr} | `{sha256}` |\n")
    (base / CHECKSUMS_NAME).write_text("".join(md), encoding="utf-8")


def append_manifest_jsonl(base: Path, rows: list[dict]):
    mf = base / MANIFEST_NAME
    with mf.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "path": r["path_rel"],
                "size_bytes": r["size_bytes"],
                "mtime": r["mtime"],
                "sha256": r["sha256"],
                **({"md5": r["md5"]} if "md5" in r else {}),
            }) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default=str(DEFAULT_BASE),
                    help="Base directory to scan")
    ap.add_argument("--include-md5", action="store_true",
                    help="Also compute MD5 (slower, optional)")
    ap.add_argument("--verify", action="store_true",
                    help="Verify files against existing CHECKSUMS.md")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    if not base.exists():
        print(f"[!] Base not found: {base}", file=sys.stderr)
        sys.exit(2)

    checksums_path = base / CHECKSUMS_NAME
    if args.verify:
        # verify mode: compare SHA256 with table
        ref = load_checksums_table(checksums_path)
        if not ref:
            print(
                f"[!] No {CHECKSUMS_NAME} found at {checksums_path}", file=sys.stderr)
            sys.exit(2)
        ok = 0
        bad = 0
        missing = 0
        for p in iter_files(base):
            rel = str(p.relative_to(base))
            sha = sha256_file(p)
            if rel not in ref:
                print(f"[MISSING IN TABLE] {rel}")
                missing += 1
                continue
            sha_ref, _, _ = ref[rel]
            if sha_ref == sha:
                ok += 1
            else:
                print(
                    f"[MISMATCH] {rel}\n  expected: {sha_ref}\n  actual:   {sha}")
                bad += 1
        print(
            f"[=] Verify summary: OK={ok}, MISMATCH={bad}, NOT_LISTED={missing}")
        sys.exit(0 if bad == 0 else 1)

    # generate mode
    # start fresh manifest
    man = base / MANIFEST_NAME
    if man.exists():
        man.unlink()

    rows_for_md: list[dict] = []
    batch_for_manifest: list[dict] = []

    total = 0
    for p in iter_files(base):
        total += 1
    print(f"[i] Files to process: {total}")

    processed = 0
    for p in iter_files(base):
        processed += 1
        rel = str(p.relative_to(base))
        st = p.stat()
        size = st.st_size
        mtime = int(st.st_mtime)
        sha = sha256_file(p)
        row = {
            "path_rel": rel,
            "size_bytes": size,
            "size_hr": human_bytes(size),
            "mtime": mtime,
            "sha256": sha,
        }
        if args.include_md5:
            row["md5"] = md5_file(p)

        rows_for_md.append(row)
        batch_for_manifest.append(row)

        # write manifest in chunks to avoid keeping everything in RAM
        if len(batch_for_manifest) >= 200:
            append_manifest_jsonl(base, batch_for_manifest)
            batch_for_manifest.clear()

        if processed % 50 == 0:
            print(f"[=] {processed}/{total} ...")

    # flush remainder
    if batch_for_manifest:
        append_manifest_jsonl(base, batch_for_manifest)

    # stable ordering in CHECKSUMS.md (alphabetical)
    rows_for_md.sort(key=lambda r: r["path_rel"])
    write_checksums_md(base, rows_for_md, include_md5=args.include_md5)

    print(f"[✓] Wrote {CHECKSUMS_NAME} and {MANIFEST_NAME} at {base}")


if __name__ == "__main__":
    main()
