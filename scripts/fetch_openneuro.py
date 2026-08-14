#!/usr/bin/env python3
"""Fetch an OpenNeuro dataset over plain HTTPS. Standard library only.

No AWS CLI, no datalad, no openneuro-py: the OpenNeuro S3 bucket is public, so
the REST listing and the objects themselves are ordinary HTTPS GETs.

    python scripts/fetch_openneuro.py ds003620 ~/scratch/ds003620
    python scripts/fetch_openneuro.py ds003620 ~/scratch/ds003620 --verify-only

Two properties matter more than speed here:

* **Every file is checked against the size in the bucket listing.** A partial
  transfer leaves a file that exists and passes every presence check, and only
  surfaces later as an unreadable recording -- or as a subject that processes to
  a plausible but wrong result.
* **Downloads land atomically.** Each object is written to a ``.part`` file and
  renamed only once its length matches. An interrupted run therefore never
  leaves a truncated file that looks complete, so re-running is always safe.
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET = "https://openneuro.org.s3.amazonaws.com"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _get(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "eeg-pipeline-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
        return resp.read()


def list_objects(dataset: str, timeout: float = 60.0) -> dict[str, int]:
    """Return {key relative to the dataset: size in bytes} for every object.

    The listing is paginated at 1000 keys; a dataset of a few hundred subjects
    runs to tens of pages, so the continuation token has to be followed.
    """
    prefix = f"{dataset}/"
    out: dict[str, int] = {}
    token: str | None = None
    page = 0
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        xml = _get(f"{BUCKET}/?{urllib.parse.urlencode(params)}", timeout=timeout)
        root = ET.fromstring(xml)
        for node in root.findall(f"{S3_NS}Contents"):
            key = node.findtext(f"{S3_NS}Key") or ""
            size = int(node.findtext(f"{S3_NS}Size") or 0)
            if key.endswith("/"):
                continue  # directory placeholder
            out[key[len(prefix):]] = size
        page += 1
        print(f"\r  listing... {len(out)} objects ({page} page(s))", end="", file=sys.stderr)
        if (root.findtext(f"{S3_NS}IsTruncated") or "false").lower() != "true":
            break
        token = root.findtext(f"{S3_NS}NextContinuationToken")
        if not token:
            break
    print(file=sys.stderr)
    if not out:
        raise SystemExit(f"No objects found for dataset {dataset!r}. Check the accession number.")
    return out


def download_one(dataset: str, key: str, size: int, dest: Path, attempts: int, timeout: float) -> str | None:
    """Download one object atomically. Returns an error string, or None on success."""
    target = dest / key
    if target.exists() and target.stat().st_size == size:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    url = f"{BUCKET}/{urllib.parse.quote(f'{dataset}/{key}')}"

    last = "unknown error"
    for attempt in range(1, attempts + 1):
        try:
            data = _get(url, timeout=timeout)
            if len(data) != size:
                # Short read: treat as a failure rather than writing it, which is
                # exactly how a truncated file would otherwise be created.
                last = f"size mismatch (got {len(data)}, expected {size})"
                raise OSError(last)
            part.write_bytes(data)
            part.replace(target)  # atomic within a filesystem
            return None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(min(2 ** attempt, 30))
        finally:
            if part.exists():
                part.unlink(missing_ok=True)
    return last


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="Accession, e.g. ds003620")
    ap.add_argument("target", type=Path, help="Destination directory")
    ap.add_argument("--attempts", type=int, default=4, help="Retries per file (default 4).")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--verify-only", action="store_true",
                    help="Report what is missing or the wrong size; download nothing.")
    ap.add_argument("--include", default=None,
                    help="Only keys containing this substring (e.g. sub-01, or .json).")
    args = ap.parse_args(argv)

    dest = args.target.expanduser().resolve()
    print(f"Dataset : {args.dataset}\nTarget  : {dest}", file=sys.stderr)

    remote = list_objects(args.dataset, timeout=args.timeout)
    if args.include:
        remote = {k: v for k, v in remote.items() if args.include in k}
        print(f"  filtered to {len(remote)} object(s) matching {args.include!r}", file=sys.stderr)

    total_bytes = sum(remote.values())
    print(f"  {len(remote)} objects, {total_bytes / 1e9:.2f} GB", file=sys.stderr)

    missing, wrong_size, present = [], [], 0
    for key, size in remote.items():
        p = dest / key
        if not p.exists():
            missing.append(key)
        elif p.stat().st_size != size:
            wrong_size.append(key)
        else:
            present += 1

    print(f"\n  complete    : {present}", file=sys.stderr)
    print(f"  missing     : {len(missing)}", file=sys.stderr)
    print(f"  wrong size  : {len(wrong_size)}   <- truncated by an earlier transfer",
          file=sys.stderr)

    if args.verify_only:
        for key in (missing + wrong_size)[:40]:
            print(f"    {key}")
        if len(missing) + len(wrong_size) > 40:
            print(f"    ... and {len(missing) + len(wrong_size) - 40} more")
        return 0 if not (missing or wrong_size) else 1

    todo = missing + wrong_size
    if not todo:
        print("\nNothing to do: every file present at the expected size.", file=sys.stderr)
        return 0

    print(f"\nFetching {len(todo)} file(s)...", file=sys.stderr)
    failures: dict[str, str] = {}
    for i, key in enumerate(todo, 1):
        err = download_one(args.dataset, key, remote[key], dest, args.attempts, args.timeout)
        if err:
            failures[key] = err
        done = i - len(failures)
        print(f"\r  {i}/{len(todo)}  ok={done} failed={len(failures)}", end="", file=sys.stderr)
    print(file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} file(s) failed:", file=sys.stderr)
        for key, err in list(failures.items())[:20]:
            print(f"  {key}\n      {err}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more", file=sys.stderr)
        # Distinguishing a stuck set from a flaky one is what tells you whether
        # re-running is worth anything.
        print("\nRe-run to retry. If the count is identical next time, the failures are "
              "deterministic rather than transient -- inspect the errors above.", file=sys.stderr)
        return 1

    print("\nAll files present at the expected size.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
