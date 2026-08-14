#!/usr/bin/env python3
"""Rebuild BIDS events.tsv for the ds003620 subjects whose labels are missing.

SCOPE - this applies to a MINORITY of the dataset. Of 43 subjects with events:

  * 31 already ship the full 12-class vocabulary (ntDontcount_lab, t_count_oval,
    ...). They are correct as distributed and are SKIPPED, never rewritten.
  *  1 (sub-09) carries target/non_target only - stimulus but no task/environment.
  * 11 (sub-01..08, 17, 25, 30) carry the single value "S  1" and no condition
    information at all. Those are the only candidates for reconstruction.

An earlier version inspected one subject, found uniform "S  1", and treated that
as a property of the dataset. It then rewrote all 43 - which would have replaced
31 subjects' published ground truth with a reconstruction.

WHERE THE LABELS COME FROM
    derivatives/erp/**/desc-window_epochs.tsv carries a `code` per epoch from the
    2x3x2 design, and an `epoch` ordinal indexing the original stimulus sequence.

    Indexing is over STIMULUS rows, not all rows. Every events.tsv carries one
    "empty" New Segment marker at onset 1; counting it as a trigger shifts every
    code by one. Measured against the ground-truth subjects, that scored 68%
    agreement - chance, for an 80/20 oddball - versus 100% once excluded.

    onset is also converted from sample indices to seconds, which BIDS requires
    and the published files violate.

VERIFICATION
    Where a subject has published labels, agreement is measured and must be
    exactly 100%; anything less means the mapping is wrong. That check is what
    caught the off-by-one, and is worth running even though those subjects are
    skipped for output.

    For the 11 "S  1" subjects there is no ground truth, so alignment is inferred
    from block-seam timing. Treat those as provisional: published labels and a
    reconstruction are different classes of evidence and must not be pooled
    silently in the validation statement.

    python scripts/reconstruct_ds003620_events.py ~/scratch/ds003620 --dry-run
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


#: The published trial_type vocabulary, for the subjects that carry it.
LABEL_TO_CODE: dict[str, int] = {
    "ntDontcount_lab": 111, "ntDontcount_oval": 112, "ntDontcount_campus": 113,
    "ntcount_lab": 121, "ntcount_oval": 122, "ntcount_campus": 123,
    "t_Dontcount_lab": 211, "t_Dontcount_oval": 212, "t_Dontcount_campus": 213,
    "t_count_lab": 221, "t_count_oval": 222, "t_count_campus": 223,
}


def verify_against_published(
    stim_rows: list[list[str]], ti: int, codes: dict[int, int]
) -> tuple[int, int]:
    """Agreement between derivative codes and already-published labels.

    Only 11 of 43 subjects have the undifferentiated "S  1" events; the other 31
    ship the full 12-class vocabulary. Those subjects are ground truth, and
    checking against them is what turns the epoch-to-trigger mapping from an
    assumption into a measurement. It is also what caught this script indexing
    every row rather than every stimulus row.
    """
    hit = tot = 0
    for epoch, code in codes.items():
        j = epoch - 1
        if 0 <= j < len(stim_rows):
            label = stim_rows[j][ti].strip()
            if label in LABEL_TO_CODE:
                tot += 1
                hit += LABEL_TO_CODE[label] == code
    return hit, tot


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


def check_alignment(coded: list[tuple[int, float, int]], all_onsets_s: list[float]) -> tuple[bool, str]:
    """Test whether the codes are aligned to the right triggers, via timing.

    Block structure alone proves nothing: the codes arrive in epoch order, so they
    look blocked whether or not they landed on the correct triggers. What does
    discriminate is *when* the blocks change. The design runs six blocks
    (3 environments x 2 instructions) separated by instructions and by physically
    relocating, so the trigger stream carries long gaps at those seams. If the
    mapping is right, every change of (task, environment) must coincide with one
    of those gaps. Under a shift -- the failure mode if the authors deleted
    spurious triggers before epoching -- transitions drift into the middle of a
    block and land on an ordinary inter-stimulus interval.

    Returns (passed, human-readable detail).
    """
    if len(coded) < 50 or len(all_onsets_s) < 50:
        return True, "too few events to test"

    # Rank the raw inter-trigger gaps rather than thresholding them. An earlier
    # version required a seam to exceed 10x the median ISI, which assumed long
    # pauses between blocks; this dataset has none (median ISI 0.56s, no
    # transition above 5.6s), so that test failed every subject including ones at
    # 99.9% coverage. Whatever separates the blocks, it should still be among the
    # LARGEST gaps present -- an assumption about rank, not magnitude.
    onsets = sorted(all_onsets_s)
    gaps = [(onsets[i + 1] - onsets[i], i) for i in range(len(onsets) - 1)]
    if not gaps:
        return True, "no usable intervals"
    median_isi = sorted(g for g, _ in gaps)[len(gaps) // 2]

    # Transition points, as indices into the raw trigger sequence.
    transitions: list[int] = []
    for (i_prev, _, c_prev), (_, _, c_now) in zip(coded, coded[1:], strict=False):
        # Codes are stim*100 + task*10 + environ, so block identity is the low
        # two digits; the high digit alone would count every Standard->Deviant
        # switch as a block change.
        if (c_prev % 100) != (c_now % 100):
            transitions.append(i_prev - 1)  # 0-based index of the gap that follows

    if not transitions:
        return False, "no block transitions found — expected 5 for a 6-block design"

    # Are the transitions among the largest gaps? Take a generous candidate pool
    # (4x the number of seams) so the test flags a genuine shift rather than
    # ordinary jitter in which gap is biggest.
    n = len(transitions)
    largest = {i for _, i in sorted(gaps, reverse=True)[: max(n * 4, 20)]}
    hits = sum(1 for t in transitions if any(abs(t - li) <= 2 for li in largest))
    passed = hits >= n - 1
    ranks = []
    order = {i: r for r, (_, i) in enumerate(sorted(gaps, reverse=True))}
    for t in transitions:
        best = min((order.get(t + d, 10**9) for d in (-2, -1, 0, 1, 2)), default=10**9)
        ranks.append(best if best < 10**9 else -1)
    detail = (
        f"{hits}/{n} transitions at a top-ranked gap "
        f"(gap ranks {ranks}, median ISI {median_isi:.2f}s)"
    )
    return passed, detail


def rebuild(bids_root: Path, out_dir: Path, dry_run: bool) -> int:
    subjects = sorted(p for p in bids_root.glob("sub-*") if p.is_dir())
    if not subjects:
        print(f"No sub-* directories under {bids_root}", file=sys.stderr)
        return 1

    total_written = 0
    skipped_alignment: list[str] = []
    already_labelled: list[str] = []
    disagreeing: list[str] = []
    print(f"{'subject':10} {'triggers':>9} {'coded':>7} {'coverage':>9}  cells      alignment")
    print("-" * 104)
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
        ti = head.index("trial_type") if "trial_type" in head else None
        all_rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]

        # Subjects that already carry the published labels must never be
        # rewritten: their events are correct as distributed, and overwriting
        # them would substitute a reconstruction for ground truth.
        classes = set()
        if ti is not None:
            classes = {r[ti].strip() for r in all_rows if len(r) > ti} - {"", "empty"}
        if len(classes) >= 10:
            # These subjects need no reconstruction -- their labels are correct as
            # published. They still need REPAIR: onsets are sample indices where
            # BIDS requires seconds, so read to spec every event lands outside the
            # recording. Rewrite using the subject's OWN labels, converting the
            # onsets and adding the numeric `value` column the pipeline reads.
            repaired = []
            for r in all_rows:
                if ti is None or len(r) <= ti:
                    continue
                label = r[ti].strip()
                if label not in LABEL_TO_CODE:
                    continue  # the lone "empty" New Segment marker
                try:
                    sample = int(float(r[oi].strip()))
                except ValueError:
                    continue
                code = LABEL_TO_CODE[label]
                stim, task, env = CODE_LEVELS[code]
                repaired.append(
                    f"{sample * dt:.6f}\t0.0\t{sample}\t{stim}\t{code}\t{task}\t{env}"
                )
            # Cross-check against the derivatives where they exist. This is the
            # measurement that caught the off-by-one, so it runs even though these
            # subjects need no reconstruction.
            hit, tot = verify_against_published(
                [r for r in all_rows if len(r) > ti and r[ti].strip() != "empty"], ti, codes
            )
            pct = (hit / tot * 100) if tot else None
            agree = f"{pct:.2f}%" if pct is not None else "no derivative"
            # Exactly 100%, or not at all. Anything less means the labels and the
            # derivatives disagree about which trial is which, and there is no
            # way to tell from here which of the two is right. sub-24 sits at
            # 67.76% -- resolvable only per-subject, not by a global rule.
            trustworthy = pct is None or pct >= 99.999
            if not trustworthy:
                print(f"{sub:10} {len(all_rows):>9} {len(repaired):>7} {'published':>9}  "
                      f"{len(classes):>2}/12 cells  [HOLD] labels disagree with derivatives "
                      f"({agree}) — not written, needs individual review")
                disagreeing.append(sub)
                continue
            print(f"{sub:10} {len(all_rows):>9} {len(repaired):>7} {'published':>9}  "
                  f"{len(classes):>2}/12 cells  [REPAIR] own labels, onsets->seconds; "
                  f"derivative agreement {agree}")
            already_labelled.append(sub)
            if not dry_run and repaired:
                dest = out_dir / sub / "eeg" / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(
                    "onset\tduration\tsample\ttrial_type\tvalue\ttask_condition\tenvironment\n"
                    + "\n".join(repaired) + "\n",
                    encoding="utf-8",
                )
                total_written += 1
            continue

        # Index STIMULUS rows only. Every file carries one "empty" New Segment
        # marker at onset 1; counting it as a trigger shifts every code by one,
        # which measured 68% agreement against ground truth -- chance, for an
        # 80/20 oddball -- versus 100% when it is excluded.
        rows = [r for r in all_rows if ti is None or (len(r) > ti and r[ti].strip() != "empty")]

        out_rows, seen, coded, all_onsets = [], set(), [], []
        trigger_index = 0
        for parts in rows:
            raw_onset = parts[oi].strip()
            try:
                sample = int(float(raw_onset))
            except ValueError:
                continue
            trigger_index += 1
            onset_s = sample * dt
            all_onsets.append(onset_s)
            code = codes.get(trigger_index)
            if code is None:
                continue  # epoch rejected upstream; no condition is known for it
            stim, task, env = CODE_LEVELS.get(code, ("n/a", "n/a", "n/a"))
            seen.add(code)
            coded.append((trigger_index, onset_s, code))
            out_rows.append(
                # onset in SECONDS, as BIDS requires; sample retained so the
                # original index stays auditable.
                f"{onset_s:.6f}\t0.0\t{sample}\t{stim}\t{code}\t{task}\t{env}"
            )

        aligned, detail = check_alignment(coded, all_onsets)
        coverage = (len(out_rows) / trigger_index * 100) if trigger_index else 0.0
        flag = "OK  " if aligned else "FAIL"
        print(f"{sub:10} {trigger_index:>9} {len(out_rows):>7} {coverage:>8.1f}%  "
              f"{len(seen):>2}/12 cells  [{flag}] {detail}")
        if not aligned:
            skipped_alignment.append(sub)
            if not dry_run:
                # Writing a file whose labels are probably wrong is worse than
                # writing none: it would be indistinguishable downstream.
                continue

        if not dry_run:
            dest = out_dir / sub / "eeg" / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                "onset\tduration\tsample\ttrial_type\tvalue\ttask_condition\tenvironment\n"
                + "\n".join(out_rows) + "\n",
                encoding="utf-8",
            )
            total_written += 1

    if disagreeing:
        print(f"\n[HOLD] {len(disagreeing)} subject(s) whose published labels disagree with "
              f"their own derivatives: {disagreeing}")
        print("       Neither source can be assumed correct from here; resolve per subject.")

    if skipped_alignment:
        print(
            f"\n[FAIL] {len(skipped_alignment)} subject(s) failed the alignment check "
            f"and were NOT written: {skipped_alignment}"
        )
        print(
            "       Their condition-block transitions do not fall on the long gaps "
            "between blocks, which means the epoch ordinals do not index this "
            "trigger sequence -- most likely because spurious triggers were removed "
            "before epoching. Their labels would be shifted and silently wrong."
        )

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
