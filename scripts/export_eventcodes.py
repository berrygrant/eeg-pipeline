#!/usr/bin/env python3
"""Export EventCode-only CSVs for each raw EEG participant."""
from __future__ import annotations

import argparse
from pathlib import Path

from eeg_pipeline.behavior import (
    resolve_subject_csv_path,
    subject_number_from_stem,
    write_eventcodes_csv,
)


def _raw_files(raw_dir: Path) -> list[Path]:
    files = sorted(p for p in raw_dir.glob("*.vhdr") if p.is_file())
    if files:
        return files
    return sorted(p for p in raw_dir.glob("*.set") if p.is_file())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export EventCode-only CSVs for each raw participant.")
    ap.add_argument("--raw_dir", required=True, help="Folder containing raw EEG files (.vhdr or .set).")
    ap.add_argument("--behavioral_dir", required=True, help="Folder containing subject-###.csv behavioral exports.")
    ap.add_argument("--out_dir", required=True, help="Destination folder for participant-eventcodes CSVs.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing participant-eventcodes files.")
    args = ap.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    behavioral_dir = Path(args.behavioral_dir)
    out_dir = Path(args.out_dir)

    raws = _raw_files(raw_dir)
    if not raws:
        raise RuntimeError(f"No .vhdr or .set files found in {raw_dir}")

    n_written = 0
    n_skipped_existing = 0
    n_missing_behavior = 0

    for raw_path in raws:
        stem = raw_path.stem
        subj_num = subject_number_from_stem(stem)
        subject_csv = resolve_subject_csv_path(behavioral_dir, subj_num, stem)
        out_csv = out_dir / f"{stem}-eventcodes.csv"

        if not subject_csv.exists():
            print(f"[WARN] Missing behavioral CSV for {stem}: {subject_csv}")
            n_missing_behavior += 1
            continue

        if out_csv.exists() and not args.overwrite:
            print(f"[SKIP] Exists: {out_csv}")
            n_skipped_existing += 1
            continue

        n_codes = write_eventcodes_csv(subject_csv, out_csv)
        n_written += 1
        print(f"[OK] {stem}: wrote {n_codes} EventCode rows -> {out_csv}")

    print(
        "\nSummary: "
        f"written={n_written}, "
        f"skipped_existing={n_skipped_existing}, "
        f"missing_behavior={n_missing_behavior}, "
        f"raw_total={len(raws)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
