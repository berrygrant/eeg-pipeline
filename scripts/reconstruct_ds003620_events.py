#!/usr/bin/env python3
"""Rebuild BIDS-compliant events.tsv for ds003620 from its published derivatives.

ds003620's events.tsv is unusable as published, in two independent ways:

* ``trial_type`` is the single value ``S  1`` for every event, so no condition
  contrast can be derived. The original ``.vmrk`` markers are identical, so the
  trigger stream never carried condition identity -- this is what the source
  workflow's "manual trigger cleanup" was reconstructing.
* ``onset`` holds sample indices, where BIDS requires seconds. Read to spec,
  every event lands far outside the recording.

The dataset's own ``derivatives/erp/**/desc-window_epochs.tsv`` carries the
answer: a ``code`` per epoch drawn from the 2x3x2 design (stimulus x environment
x instruction), and an ``epoch`` ordinal that is the position within the *original*
trigger sequence -- verified by the ordinals having gaps where epochs were
rejected (sub-01: 3682 distinct values reaching 4494). So epoch N identifies
trigger N, and the codes can be attached to the trigger onsets.

    python scripts/reconstruct_ds003620_events.py ~/scratch/ds003620 --out-dir ~/scratch/ds003620_events

Writes a mirror of ``sub-*/eeg/*_events.tsv`` under --out-dir rather than editing
the dataset, which is a datalad clone. Apply them with the printed command once
the coverage report looks right.

WHAT THIS INHERITS
    Only triggers that survived the authors' epoch rejection carry a code, so a
    reconstructed file covers ~82% of triggers for sub-01. Labelling is therefore
    downstream of their rejection decisions, even though this pipeline still
    applies its own artifact rejection afterwards. That is a real limitation to
    record in the validation statement: it is not a fully independent analysis.
    Reconstructing from sourcedata/sub-N/behav/subject-N.csv would avoid it, and
    can be checked against this output, which is authoritative where it exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# From derivatives/erp/**/desc-window_epochs.json "code" Levels.
CODE_LEVELS: dict[int, tuple[str, str, str]] = {
    111: ("Standard", "Ignore", "Lab"),
    112: ("Standard", "Ignore", "Oval"),
    113: ("Standard", "Ignore", "Campus"),
    121: ("Standard", "Count", "Lab"),
    122: ("Standard", "Count", "Oval"),
    123: ("Standard", "Count", "Campus"),
    211: ("Deviant", "Ignore", "Lab"),
    212: ("Deviant", "Ignore", "Oval"),
    213: ("Deviant", "Ignore", "Campus"),
    221: ("Deviant", "Count", "Lab"),
    222: ("Deviant", "Count", "Oval"),
    223: ("Deviant", "Count", "Campus"),
}


def sampling_interval_s(vhdr: Path) -> float:
    """Seconds per sample, from the BrainVision header."""
    for line in vhdr.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("SamplingInterval="):
            return float(line.split("=", 1)[1]) / 1e6
    raise ValueError(f"No SamplingInterval in {vhdr}")


def epoch_codes(epochs_tsv: Path) -> dict[int, int]:
    """{epoch ordinal: trigger code}, collapsed over channels."""
    lines = epochs_tsv.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    ci, ei = head.index("code"), head.index("epoch")
    out: dict[int, int] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        try:
            out[int(parts[ei])] = int(parts[ci])
        except (ValueError, IndexError):
            continue
    return out


def rebuild(bids_root: Path, out_dir: Path, dry_run: bool) -> int:
    subjects = sorted(p for p in bids_root.glob("sub-*") if p.is_dir())
    if not subjects:
        print(f"No sub-* directories under {bids_root}", file=sys.stderr)
        return 1

    total_written = 0
    print(f"{'subject':10} {'triggers':>9} {'coded':>7} {'coverage':>9}  conditions")
    print("-" * 72)
    for sub_dir in subjects:
        sub = sub_dir.name
        events = sorted(sub_dir.glob("eeg/*_events.tsv"))
        vhdrs = sorted(sub_dir.glob("eeg/*.vhdr"))
        epochs = sorted((bids_root / "derivatives" / "erp" / sub / "eeg").glob("*desc-window_epochs.tsv"))
        if not (events and vhdrs and epochs):
            print(f"{sub:10} {'-':>9} {'-':>7} {'-':>9}  missing events/vhdr/derivatives")
            continue

        try:
            dt = sampling_interval_s(vhdrs[0])
            codes = epoch_codes(epochs[0])
        except Exception as exc:
            print(f"{sub:10} {'-':>9} {'-':>7} {'-':>9}  {type(exc).__name__}: {exc}")
            continue

        src = events[0]
        lines = src.read_text(encoding="utf-8").splitlines()
        head = lines[0].split("\t")
        oi = head.index("onset")
        rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]

        out_rows, seen = [], set()
        trigger_index = 0
        for parts in rows:
            raw_onset = parts[oi].strip()
            try:
                sample = int(float(raw_onset))
            except ValueError:
                continue
            trigger_index += 1
            code = codes.get(trigger_index)
            if code is None:
                continue  # epoch rejected upstream; no condition is known for it
            stim, task, env = CODE_LEVELS.get(code, ("n/a", "n/a", "n/a"))
            seen.add(code)
            out_rows.append(
                # onset in SECONDS, as BIDS requires; sample retained so the
                # original index stays auditable.
                f"{sample * dt:.6f}\t0.0\t{sample}\t{stim}\t{code}\t{task}\t{env}"
            )

        coverage = (len(out_rows) / trigger_index * 100) if trigger_index else 0.0
        print(f"{sub:10} {trigger_index:>9} {len(out_rows):>7} {coverage:>8.1f}%  "
              f"{len(seen)}/12 cells present")

        if not dry_run:
            dest = out_dir / sub / "eeg" / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                "onset\tduration\tsample\ttrial_type\tvalue\ttask_condition\tenvironment\n"
                + "\n".join(out_rows) + "\n",
                encoding="utf-8",
            )
            total_written += 1

    if not dry_run:
        print(f"\nWrote {total_written} events file(s) under {out_dir}")
        print("Apply them over a WRITABLE copy of the dataset with:")
        print(f"  rsync -av {out_dir}/ <writable-copy-of-ds003620>/")
        print("Do not write into the datalad clone: its events.tsv files are tracked in git.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bids_root", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Report coverage only.")
    args = ap.parse_args(argv)

    bids_root = args.bids_root.expanduser().resolve()
    out_dir = (args.out_dir or (bids_root.parent / f"{bids_root.name}_events")).expanduser().resolve()
    if not args.dry_run:
        print(f"Reconstructing events for {bids_root}\n  -> {out_dir}\n")
    return rebuild(bids_root, out_dir, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
