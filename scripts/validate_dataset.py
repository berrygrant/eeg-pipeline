#!/usr/bin/env python3
"""Run one dataset through the validation protocol's gates and package the evidence.

The pipeline emits per-subject derivatives and combined tables, but not the
provenance and reconciliation artifacts a validation package needs: there is
nothing recording the code commit, the environment, or a config hash, and nothing
that reconciles participant counts from the dataset through QC to metrics. This
wraps a run and produces those.

    python scripts/validate_dataset.py \
        --config config.yaml \
        --bids-root /path/to/ds003620 \
        --out-dir  /path/to/runs/ds003620_2026-08-12 \
        --dataset-id ds003620 --dataset-version 1.1.1

Gates implemented here:
  0  freeze provenance    -> run_manifest.json
  1  dataset integrity    -> dataset_integrity.json  (refuses to continue on error
                             unless --allow-integrity-warnings)
  2  run into an empty destination, capturing the full log -> pipeline_run.log
  3  reconcile QC         -> participant_flow.csv
  5  package the evidence -> validation_package/ + validation_statement.md stub

Gate 4 (signal validation against a published benchmark) is deliberately not
automated: choosing contrasts, outlier rules and the claim level is the analyst's
judgment, and a script that guessed would produce exactly the unfounded numbers
the protocol is written to prevent. The stub records where those go.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Gate 0 - provenance
# ---------------------------------------------------------------------------
def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build_run_manifest(args, config_text: str, started_at: str) -> dict:
    versions: dict[str, str] = {"python": platform.python_version()}
    for mod in ("mne", "numpy", "scipy", "pandas", "pyarrow", "matplotlib"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "not installed"
    try:
        from eeg_pipeline import __version__ as pipeline_version
    except Exception:
        pipeline_version = "unknown"

    dirty = bool(_git("status", "--porcelain"))
    return {
        "Dataset": {
            "id": args.dataset_id,
            "version": args.dataset_version,
            "bids_root": str(args.bids_root),
            "retrieved": args.retrieved or "not recorded",
        },
        "Code": {
            "pipeline_version": pipeline_version,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_describe": _git("describe", "--tags", "--always", "--dirty"),
            # A dirty tree corresponds to no commit at all, so the run is not
            # reproducible from the recorded identifiers alone. Recorded rather
            # than blocked, but the packaging step warns on it.
            "git_dirty": dirty,
        },
        "Config": {
            "path": str(args.config),
            "sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        },
        "Environment": versions,
        "Platform": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        },
        "Execution": {"started_at": started_at, "ended_at": None, "argv": list(sys.argv)},
    }


# ---------------------------------------------------------------------------
# Gate 1 - dataset integrity
# ---------------------------------------------------------------------------
def _normalize_label(label: str) -> str:
    """Compare subject labels ignoring zero-padding (sub-1 vs sub-01)."""
    body = label[4:] if label.startswith("sub-") else label
    return f"sub-{body.lstrip('0') or '0'}" if body.isdigit() else label


def check_dataset_integrity(bids_root: Path, subjects: list[str] | None = None) -> dict:
    report: dict = {"errors": [], "warnings": [], "counts": {}}
    if not bids_root.is_dir():
        report["errors"].append(f"BIDS root does not exist: {bids_root}")
        return report

    subject_dirs = sorted(p.name for p in bids_root.glob("sub-*") if p.is_dir())
    report["counts"]["subject_directories"] = len(subject_dirs)

    # When a subset is requested, per-file checks are scoped to it. Otherwise a
    # deliberate two-subject smoke test fails on content never fetched for the
    # other forty-two.
    if subjects:
        wanted = {s if s.startswith("sub-") else f"sub-{s}" for s in subjects}
        scoped = sorted(d for d in subject_dirs if d in wanted or _normalize_label(d) in
                        {_normalize_label(w) for w in wanted})
        report["counts"]["subjects_checked"] = len(scoped)
        unknown = sorted(wanted - set(subject_dirs))
        if unknown:
            report["errors"].append(f"requested subject(s) not present: {unknown}")
    else:
        scoped = subject_dirs

    participants = bids_root / "participants.tsv"
    if not participants.exists():
        report["warnings"].append("participants.tsv is missing")
    else:
        rows = [ln for ln in participants.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]
        listed = {ln.split("\t")[0].strip() for ln in rows}
        report["counts"]["participants_tsv_rows"] = len(rows)
        missing_dir = sorted(listed - set(subject_dirs))
        missing_row = sorted(set(subject_dirs) - listed)

        # Distinguish "the same subjects, written differently" from "subjects are
        # genuinely absent". Zero-padding drift is a real defect -- a join on
        # participant_id silently drops those rows -- but it does not stop the
        # pipeline, which discovers subjects from directories.
        norm_listed = {_normalize_label(x) for x in missing_dir}
        norm_dirs = {_normalize_label(x) for x in missing_row}
        padding_only = sorted(norm_listed & norm_dirs)
        if padding_only:
            report["warnings"].append(
                f"{len(padding_only)} subject(s) differ only by zero-padding between "
                f"participants.tsv and the directory names, e.g. "
                f"{[x for x in missing_dir if _normalize_label(x) in padding_only][:3]} vs "
                f"{[x for x in missing_row if _normalize_label(x) in padding_only][:3]}. "
                "Processing is unaffected (subjects are discovered from directories), "
                "but any join on participant_id will silently drop them."
            )
            report["padding_mismatch"] = padding_only
        truly_missing_dir = [x for x in missing_dir if _normalize_label(x) not in padding_only]
        truly_missing_row = [x for x in missing_row if _normalize_label(x) not in padding_only]
        if truly_missing_dir:
            report["errors"].append(f"listed in participants.tsv but no directory: {truly_missing_dir}")
        if truly_missing_row:
            report["errors"].append(f"directory present but absent from participants.tsv: {truly_missing_row}")

    raws, no_events, broken_links = [], [], []
    for sub in scoped:
        for ext in (".vhdr", ".set"):
            raws.extend(sorted((bids_root / sub).glob(f"**/eeg/*{ext}")))
    raws.sort()
    report["counts"]["raw_recordings"] = len(raws)

    for raw in raws:
        stem = raw.name
        for suffix in ("_eeg.vhdr", "_eeg.set", ".vhdr", ".set"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if not list(raw.parent.glob(f"{stem}*_events.tsv")):
            no_events.append(str(raw.relative_to(bids_root)))
        # A BrainVision header points at .eeg/.vmrk by name; a missing target is
        # the "broken BrainVision link" the handoff records for one participant.
        if raw.suffix == ".vhdr":
            try:
                text = raw.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:  # pragma: no cover - defensive
                broken_links.append(f"{raw.name}: unreadable ({exc})")
                continue
            for line in text.splitlines():
                if line.startswith(("DataFile=", "MarkerFile=")):
                    target = raw.parent / line.split("=", 1)[1].strip()
                    if not target.exists():
                        broken_links.append(f"{raw.name} -> missing {target.name}")

    # Empty or implausibly small files are the signature of an interrupted
    # download: the path exists, so every presence check above passes, and the
    # failure only surfaces later as an unreadable recording mid-run.
    empty, tiny, dangling = [], [], []
    scanned = [q for sub in scoped for q in sorted((bids_root / sub).glob("**/*"))]
    for path in scanned:
        # A git-annex clone without content is the common case here: OpenNeuro's
        # GitHub mirrors are annex repos, so `git clone` alone yields symlinks
        # into .git/annex with nothing behind them. They are not files, so every
        # size check below would skip them and the dataset would look complete.
        if path.is_symlink() and not path.exists():
            dangling.append(str(path.relative_to(bids_root)))
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size == 0:
            empty.append(str(path.relative_to(bids_root)))
        elif path.suffix.lower() in {".eeg", ".fdt", ".set", ".bdf", ".edf"} and size < 4096:
            # A real continuous-EEG data file is never a few hundred bytes.
            tiny.append(f"{path.relative_to(bids_root)} ({size} B)")
    # BIDS requires events.tsv onsets in SECONDS. ds003620 publishes sample
    # indices, so anything reading the file to spec places every event far
    # outside the recording -- silently, since nothing about the numbers looks
    # wrong on its own. Checked against the BrainVision header, which states the
    # sample count and interval directly, so this needs no EEG library.
    onset_issues, flat_trial_types = [], []
    for raw in raws:
        if raw.suffix.lower() != ".vhdr":
            continue
        try:
            header = raw.read_text(encoding="utf-8", errors="replace")
            n_points = int(next(ln.split("=", 1)[1] for ln in header.splitlines()
                                if ln.startswith("DataPoints=")))
            interval_us = float(next(ln.split("=", 1)[1] for ln in header.splitlines()
                                     if ln.startswith("SamplingInterval=")))
            duration_s = n_points * interval_us / 1e6
        except Exception:
            continue
        for ev in raw.parent.glob(f"{raw.name.split('_eeg')[0]}*_events.tsv"):
            try:
                lines = ev.read_text(encoding="utf-8").splitlines()
                head = lines[0].split("\t")
                rows_ = [ln.split("\t") for ln in lines[1:] if ln.strip()]
                if not rows_ or "onset" not in head:
                    continue
                oi = head.index("onset")
                onsets = [float(r[oi]) for r in rows_ if len(r) > oi and r[oi].strip()]
                if onsets and max(onsets) > duration_s * 1.01:
                    onset_issues.append(
                        f"{ev.relative_to(bids_root)}: max onset {max(onsets):.0f} exceeds the "
                        f"{duration_s:.1f}s recording — onsets appear to be SAMPLES, not seconds"
                    )
                if "trial_type" in head:
                    ti = head.index("trial_type")
                    labels = {r[ti].strip() for r in rows_ if len(r) > ti and r[ti].strip()}
                    if len(labels) < 2:
                        flat_trial_types.append(
                            f"{ev.relative_to(bids_root)}: trial_type has a single value "
                            f"{sorted(labels)} across {len(rows_)} events — no condition "
                            "contrast can be derived from this file"
                        )
            except Exception:
                continue
    if onset_issues:
        report["errors"].append(f"{len(onset_issues)} events file(s) with out-of-range onsets")
        report["onset_out_of_range"] = onset_issues[:20]
    if flat_trial_types:
        report["errors"].append(
            f"{len(flat_trial_types)} events file(s) carry no condition distinction"
        )
        report["undifferentiated_events"] = flat_trial_types[:20]

    if dangling:
        report["errors"].append(
            f"{len(dangling)} unresolved git-annex pointer(s) — the repository was cloned "
            "but its content was never fetched (run `datalad get .` or `git annex get .`)"
        )
        report["dangling_symlinks"] = dangling[:50]
    if empty:
        report["errors"].append(f"{len(empty)} zero-byte file(s) — likely an interrupted transfer")
        report["zero_byte_files"] = empty[:50]
    if tiny:
        report["errors"].append(f"{len(tiny)} implausibly small EEG data file(s)")
        report["truncated_files"] = tiny[:50]

    if no_events:
        report["warnings"].append(f"{len(no_events)} recording(s) with no events.tsv")
        report["recordings_without_events"] = no_events[:50]
    if broken_links:
        report["errors"].append(f"{len(broken_links)} broken BrainVision link(s)")
        report["broken_links"] = broken_links[:50]
    return report


# ---------------------------------------------------------------------------
# Gate 3 - participant flow
# ---------------------------------------------------------------------------
def build_participant_flow(derivatives_root: Path) -> tuple[list[dict], list[str]]:
    """Reconcile attempted / excluded-by-reason / successful / metric-bearing."""
    import pandas as pd

    notes: list[str] = []
    root = derivatives_root / "eeg-pipeline"
    qc_path = root / "eeg" / "desc-summary_qc.tsv"
    if not qc_path.exists():
        return [], [f"no QC summary at {qc_path}"]

    entity_cols = {c: str for c in ("subject", "session", "task", "acq", "run", "recording")}
    qc = pd.read_csv(qc_path, sep="\t", dtype=entity_cols)
    status = qc["status"].astype(str).str.strip().str.upper()

    erp_path = root / "eeg" / "desc-erp_metrics.tsv"
    metric_subjects: set[str] = set()
    if erp_path.exists():
        erp = pd.read_csv(erp_path, sep="\t", dtype={"subject": str})
        metric_subjects = set(erp["subject"].astype(str))
    else:
        notes.append("no dataset-level ERP metrics table; metric-bearing count is 0")

    rows = [{"category": "attempted", "reason": "", "n": int(len(qc))}]
    for value, n in status.value_counts().items():
        rows.append(
            {
                "category": "successful" if value == "OK" else "excluded",
                "reason": value,
                "n": int(n),
            }
        )

    # The QC table already carries the reason; making the analyst open it to see
    # why every recording failed defeats the point of reconciling here.
    failed = qc.loc[~status.eq("OK")]
    if not failed.empty and "error" in failed.columns:
        seen: dict[str, list[str]] = {}
        for _, row in failed.iterrows():
            msg = str(row.get("error") or row.get("status") or "").strip()
            if msg and msg.lower() != "nan":
                seen.setdefault(msg, []).append(str(row.get("subject", "?")))
        for msg, subs in list(seen.items())[:5]:
            shown = ", ".join(subs[:4]) + (" ..." if len(subs) > 4 else "")
            notes.append(f"{len(subs)} recording(s) [{shown}]: {msg}")

    ok = qc.loc[status.eq("OK")]
    ok_subjects = set(ok["subject"].astype(str))
    bearing = ok_subjects & metric_subjects
    rows.append({"category": "metric_bearing", "reason": "", "n": int(len(bearing))})

    # The inequality the protocol asks to be explained, surfaced by name rather
    # than left as a count difference for someone to chase.
    orphans = sorted(ok_subjects - metric_subjects)
    if orphans:
        notes.append(
            f"{len(orphans)} QC-OK subject(s) contribute no metric row: {orphans[:20]}"
            + (" ..." if len(orphans) > 20 else "")
        )
        notes.append(
            "check the run log for '[WARN] ERP metrics' lines, and look for "
            "header-only per-subject desc-erp_metrics.tsv files"
        )

    total = int(status.value_counts().sum())
    if total != len(qc):
        notes.append(f"flow does not balance: statuses sum to {total}, QC has {len(qc)} rows")
    return rows, notes


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--bids-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path, help="Must be empty or absent (Gate 2).")
    ap.add_argument("--dataset-id", default="unknown")
    ap.add_argument("--dataset-version", default="unknown")
    ap.add_argument("--retrieved", default=None, help="Retrieval date, e.g. 2026-08-12.")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--allow-integrity-warnings", action="store_true")
    ap.add_argument("--skip-run", action="store_true", help="Re-package an existing run.")
    ap.add_argument(
        "--stage",
        choices=("all", "precheck", "package"),
        default="all",
        help=(
            "all: gates 0-5 in one serial process (a workstation run). "
            "precheck: gates 0-2 guard only -- writes the manifest, the integrity "
            "report and subjects.txt, then stops, so a SLURM array can process "
            "subjects independently. "
            "package: gates 3-5 -- runs --aggregate_only over whatever the array "
            "produced, then reconciles and packages."
        ),
    )
    ap.add_argument(
        "--subject-list",
        type=Path,
        default=None,
        help="Where precheck writes the subject labels (default: <out-dir>/subjects.txt).",
    )
    args = ap.parse_args(argv)

    # Resolve before anything else: the pipeline subprocess runs with cwd set to
    # the repo root, so a relative path given on the command line would quietly
    # resolve somewhere other than where the caller meant.
    args.config = args.config.expanduser().resolve()
    args.bids_root = args.bids_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if not args.config.is_file():
        print(f"[FAIL] config not found: {args.config}", file=sys.stderr)
        return 2

    started_at = datetime.now(timezone.utc).isoformat()
    out = args.out_dir
    derivatives = out / "derivatives"

    # Gate 2: an empty destination, checked before anything is written.
    if out.exists() and any(out.iterdir()) and not args.skip_run and args.stage in ("all", "precheck"):
        print(f"[FAIL] Gate 2: output directory is not empty: {out}", file=sys.stderr)
        print("       Rerun into a new directory, or pass --skip-run to re-package.", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    config_text = args.config.read_text(encoding="utf-8")

    print("== Gate 0: freeze provenance ==")
    manifest = build_run_manifest(args, config_text, started_at)
    if manifest["Code"]["git_dirty"]:
        print("[WARN] the working tree is dirty; this run matches no commit.")
    (out / "config.snapshot.yaml").write_text(config_text, encoding="utf-8")

    print("== Gate 1: dataset integrity ==")
    integrity = check_dataset_integrity(args.bids_root, subjects=args.subjects)
    (out / "dataset_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    for key in ("errors", "warnings"):
        for item in integrity[key]:
            print(f"  [{key[:-1].upper()}] {item}")
    print(f"  counts: {integrity['counts']}")
    if integrity["errors"] and not args.allow_integrity_warnings:
        print("[FAIL] Gate 1: integrity errors. Fix them, or pass "
              "--allow-integrity-warnings to proceed and document them.", file=sys.stderr)
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return 1

    if args.stage == "precheck":
        subject_list = args.subject_list or (out / "subjects.txt")
        subject_list.parent.mkdir(parents=True, exist_ok=True)
        subjects = sorted(p.name for p in args.bids_root.glob("sub-*") if p.is_dir())
        # No trailing blank line: hpc/submit.sh sizes the array by non-blank line
        # count and slurm_array.sbatch indexes non-blank lines, so the two agree
        # either way, but an exact file is easier to eyeball.
        subject_list.write_text("\n".join(subjects) + "\n", encoding="utf-8")
        manifest["Execution"]["ended_at"] = datetime.now(timezone.utc).isoformat()
        manifest["Execution"]["stage"] = "precheck"
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"  wrote {len(subjects)} subject label(s) -> {subject_list}")
        print("\nPrecheck passed. Submit the array, then run --stage package.")
        return 0

    log_path = out / "pipeline_run.log"
    if args.stage == "package":
        # The array already processed every subject with --skip_aggregate, so the
        # dataset-level tables do not exist yet. Build them from what landed.
        print("== gather: --aggregate_only ==")
        cmd = [
            sys.executable, "-m", "eeg_pipeline.cli",
            "--config", str(args.config),
            "--bids_root", str(args.bids_root),
            "--derivatives_root", str(derivatives),
            "--aggregate_only",
        ]
        print("  " + " ".join(cmd))
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n# gather: {' '.join(cmd)}\n")
            fh.flush()
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
        print(f"  exit code: {proc.returncode}")
        manifest["Execution"]["gather_exit_code"] = proc.returncode
        if proc.returncode != 0:
            print("[WARN] gather exited non-zero; the tables below may be incomplete.")
    elif not args.skip_run:
        print("== Gate 2: run ==")
        cmd = [
            sys.executable, "-m", "eeg_pipeline.cli",
            "--config", str(args.config),
            "--bids_root", str(args.bids_root),
            "--derivatives_root", str(derivatives),
            "--process_data", "--get_metrics",
            "--n_jobs", str(args.n_jobs),
        ]
        if args.subjects:
            cmd += ["--subjects", *args.subjects]
        print("  " + " ".join(cmd))
        # stderr merged into stdout so warnings and errors stay interleaved in
        # one file rather than split across two streams.
        with log_path.open("w", encoding="utf-8") as fh:
            fh.write(f"# command: {' '.join(cmd)}\n# started: {started_at}\n\n")
            fh.flush()
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
        print(f"  exit code: {proc.returncode}  (log: {log_path})")
        manifest["Execution"]["pipeline_exit_code"] = proc.returncode
        if proc.returncode != 0:
            print("[WARN] the pipeline exited non-zero; one recording's error aborts "
                  "the whole run, so later participants may never have been attempted.")
    manifest["Execution"]["stage"] = args.stage

    manifest["Execution"]["ended_at"] = datetime.now(timezone.utc).isoformat()
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("== Gate 3: reconcile QC ==")
    flow, notes = build_participant_flow(derivatives)
    if flow:
        import pandas as pd

        pd.DataFrame(flow).to_csv(out / "participant_flow.csv", index=False)
        for row in flow:
            label = f"{row['category']}{'/' + row['reason'] if row['reason'] else ''}"
            print(f"  {label:34} {row['n']}")
    for note in notes:
        print(f"  [NOTE] {note}")
    (out / "participant_flow_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print("== Gate 5: package ==")
    stub = out / "validation_statement.md"
    if not stub.exists():
        stub.write_text(
            f"""# Validation statement - {args.dataset_id} {args.dataset_version}

Generated {manifest["Execution"]["ended_at"]}. Fill in before citing this run.

## Outcome
&lt;exact replication | consistent in principle | technical feasibility&gt;

Pick from the claim ladder deliberately. "Exact replication" requires matching
processing, exclusions, timing corrections, reference, filters, metrics and
analysis to the benchmark.

## Benchmark comparison
&lt;polarity, amplitude, latency, topography vs the published result&gt;

## Deviations from the benchmark
&lt;every procedural difference, including ones judged immaterial&gt;

## Participant accounting
See participant_flow.csv. Every inequality between attempted, successful,
metric-bearing and figure-bearing needs a documented cause here.

## Outlier and robustness
&lt;rule defined BEFORE inspecting the final group estimate; report robust and
sensitivity estimates side by side&gt;

## Limitations
&lt;...&gt;

## Decision
&lt;...&gt;
""",
            encoding="utf-8",
        )
        print(f"  wrote {stub.name} (stub - Gate 4 and the claim level are yours to fill in)")

    print(f"\nPackage: {out}")
    for name in ("run_manifest.json", "config.snapshot.yaml", "dataset_integrity.json",
                 "pipeline_run.log", "participant_flow.csv", "validation_statement.md"):
        p = out / name
        print(f"  {'OK ' if p.exists() else '-- '} {name}")

    # A partial run is a legitimate result -- the flow table says who is missing.
    # No run at all is not: with no QC table there is nothing to reconcile, and
    # returning 0 here would present an empty directory as a finished package.
    if not flow:
        print(
            "\n[FAIL] no QC rows: not one recording was processed. This is a "
            "systemic failure, not per-subject exclusions -- check the array task "
            "logs (logs/validate-*_*.err) before re-running.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
