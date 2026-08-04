"""Dataset-level aggregation of per-subject derivatives.

Rebuilds the combined QC/metrics tables and grand averages by reading the
per-subject files that each processed recording writes, rather than from
in-memory state accumulated during a single run.

Reading from disk is what makes aggregation re-runnable. ``run_full_pipeline``
calls this as its tail, and ``--aggregate_only`` calls the same code after N
independent per-subject jobs have finished (e.g. one SLURM array task per
subject). The serial and array-parallel paths therefore produce identical
dataset-level outputs by construction, rather than by two implementations
happening to agree.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mne
import pandas as pd

from . import __version__
from .bids import (
    PIPELINE_NAME,
    dataset_derivative_path,
    derivative_sidecar_path,
    parse_bids_entities_like_name,
    write_json,
)
from .cli_common import _save_dataframe_with_sidecar
from .evoked import grand_averages
from .qc import write_qc_summary

#: Per-subject filename patterns consumed by the gather step.
QC_PATTERN = "*desc-summary_qc.tsv"
ERP_METRICS_PATTERN = "*desc-erp_metrics.tsv"
TFR_METRICS_PATTERN = "*desc-tfr_metrics.tsv"
ERP_TIMESERIES_PATTERN = "*desc-erp_timeseries.parquet"
EVOKED_PATTERN = "*_ave.fif"


def _subject_scoped_files(dataset_root: Path, pattern: str) -> list[Path]:
    """Return per-subject derivative files matching ``pattern``.

    Dataset-level outputs live in ``<root>/eeg/`` while per-subject outputs live
    in ``<root>/sub-XX/[ses-YY/]eeg/``. Requiring a ``sub-`` path component is
    what stops a previously written dataset-level table (or grand average) from
    being folded back into its own replacement when aggregation is re-run.
    """
    dataset_root = Path(dataset_root)
    matches: list[Path] = []
    for path in dataset_root.rglob(pattern):
        if not path.is_file():
            continue
        # Inspect directory components only: per-subject *filenames* also begin
        # with "sub-", so including the filename would classify by name rather
        # than by location and could admit a dataset-level file.
        if any(part.startswith("sub-") for part in path.relative_to(dataset_root).parent.parts):
            matches.append(path)
    return sorted(matches)


def _desc_from_stem(stem: str) -> str | None:
    """Extract the ``desc-`` value from a BIDS-like stem, if present."""
    for token in stem.split("_"):
        if token.startswith("desc-"):
            value = token[len("desc-") :]
            return value or None
    return None


#: Columns holding BIDS entity labels, which must survive a text round-trip as
#: written. Without this pandas infers a zero-padded run like "01" as the integer
#: 1, so the combined tables would disagree with the filenames and with the
#: per-subject tables they were built from.
ENTITY_COLUMNS = ("subject", "session", "task", "run")


def _recording_key(entities: Mapping[str, str]) -> tuple[str, str, str, str]:
    """Normalize BIDS entities to a comparable per-recording key.

    QC rows carry prefixed labels (``sub-01``, ``ses-02``) while filenames carry
    bare values, so both sides are stripped to bare form before comparison.
    """

    def _bare(value: Any, prefix: str) -> str:
        text = "" if value is None else str(value).strip()
        if text.lower() in {"", "nan", "none"}:
            return ""
        return text[len(prefix) + 1 :] if text.startswith(f"{prefix}-") else text

    return (
        _bare(entities.get("sub"), "sub"),
        _bare(entities.get("ses"), "ses"),
        _bare(entities.get("task"), "task"),
        _bare(entities.get("run"), "run"),
    )


def _excluded_recording_keys(dataset_root: Path) -> set[tuple[str, str, str, str]]:
    """Recordings whose most recent QC row reports something other than success.

    Aggregation rebuilds dataset-level outputs from whatever per-subject files are
    on disk, and a skipped recording's metrics/evokeds from an *earlier* run are
    not removed. Without this filter, re-running with a stricter setting (say a
    tighter ``--max_reject_rate``) would rewrite the QC row to a skip status while
    the stale metrics and evokeds were folded back into the combined tables and
    grand averages — the dataset would report a subject excluded while still
    averaging it in.

    QC rows are rewritten on every run, including skips, so they are the
    authoritative record of what the latest run decided about each recording.
    """
    excluded: set[tuple[str, str, str, str]] = set()
    for path in _subject_scoped_files(dataset_root, QC_PATTERN):
        df = _read_table(path)
        if df is None or df.empty or "status" not in df.columns:
            continue
        for row in df.to_dict("records"):
            status = str(row.get("status", "")).strip().upper()
            if not status or status == "OK":
                continue
            excluded.add(
                _recording_key(
                    {
                        "sub": row.get("subject"),
                        "ses": row.get("session"),
                        "task": row.get("task"),
                        "run": row.get("run"),
                    }
                )
            )
    return excluded


def _is_excluded(path: Path, excluded: set[tuple[str, str, str, str]] | None) -> bool:
    """Whether ``path`` belongs to a recording the latest run skipped.

    A filename that cannot be parsed is never excluded: without a key there is no
    evidence it was skipped, and dropping it would silently shrink the outputs.
    """
    if not excluded:
        return False
    try:
        return _recording_key(parse_bids_entities_like_name(path.stem)) in excluded
    except ValueError:
        return False


def _read_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix == ".parquet":
            # Parquet stores dtypes, so entity labels round-trip as written.
            return pd.read_parquet(path)
        sep = "\t" if path.suffix == ".tsv" else ","
        header = pd.read_csv(path, sep=sep, nrows=0)
        entity_dtypes = {col: str for col in ENTITY_COLUMNS if col in header.columns}
        return pd.read_csv(path, sep=sep, dtype=entity_dtypes)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[WARN] Could not read {path}: {e}")
        return None


def _concat_subject_tables(
    dataset_root: Path,
    pattern: str,
    excluded: set[tuple[str, str, str, str]] | None = None,
) -> tuple[pd.DataFrame | None, int]:
    """Concatenate every per-subject table matching ``pattern``.

    Returns the combined frame (or ``None``) and the number of files contributing
    to it, so callers can report coverage instead of silently emitting a table
    built from fewer subjects than expected. Recordings in ``excluded`` are
    skipped, so stale files left by an earlier run cannot re-enter the combined
    tables after a re-run decided to skip that recording.
    """
    frames: list[pd.DataFrame] = []
    for path in _subject_scoped_files(dataset_root, pattern):
        if _is_excluded(path, excluded):
            continue
        df = _read_table(path)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return None, 0
    return pd.concat(frames, ignore_index=True), len(frames)


def _condition_for_evoked(path: Path) -> str | None:
    """Recover the condition label for a per-subject evoked file.

    The sidecar records the condition with its original capitalization; the
    filename only carries the lower-cased ``desc-`` token. Prefer the sidecar so
    grand-average metadata matches what the serial path wrote.
    """
    sidecar = derivative_sidecar_path(path)
    if sidecar.exists():
        try:
            condition = json.loads(sidecar.read_text(encoding="utf-8")).get("Condition")
            if condition:
                return str(condition)
        except Exception:  # pragma: no cover - defensive
            pass
    return _desc_from_stem(path.stem)


def aggregate_qc_summary(dataset_root: Path) -> int:
    """Rebuild the dataset-level QC summary from per-subject QC rows."""
    df, n_files = _concat_subject_tables(dataset_root, QC_PATTERN)
    if df is None:
        return 0

    qc_path = dataset_derivative_path(dataset_root, suffix="qc", extension=".tsv", desc="summary")
    write_qc_summary(df.to_dict("records"), qc_path)
    write_json(
        derivative_sidecar_path(qc_path),
        {
            "Description": "Dataset-level QC summary for eeg-pipeline derivatives.",
            "GeneratedBy": [{"Name": PIPELINE_NAME, "Version": __version__}],
        },
    )
    return n_files


def _dataset_scoped_files(dataset_root: Path, pattern: str) -> list[Path]:
    """Return dataset-level files matching ``pattern`` (``<root>/eeg/``).

    The mirror of :func:`_subject_scoped_files`: it deliberately excludes anything
    under a ``sub-`` directory so cleanup can never reach per-subject data.
    """
    dataset_root = Path(dataset_root)
    matches: list[Path] = []
    for path in dataset_root.rglob(pattern):
        if not path.is_file():
            continue
        parts = path.relative_to(dataset_root).parent.parts
        if not any(part.startswith("sub-") for part in parts):
            matches.append(path)
    return sorted(matches)


def _remove_dataset_output(path: Path) -> None:
    """Delete a dataset-level output (and its sidecar) that has no inputs.

    Aggregation only writes outputs it can reconstruct, so without this a table or
    grand average from a previous gather survives one that found nothing to build
    it from -- and `--plot_figures` then reads it as though it belonged to the
    current run. Only dataset-level *derived* files are removed; per-subject data
    is never touched, so anything deleted here is regenerable by re-running the
    gather with its inputs present.
    """
    for candidate in (path, derivative_sidecar_path(path)):
        try:
            if candidate.exists():
                candidate.unlink()
                print(f"Removed stale dataset output (no inputs): {candidate.name}")
        except OSError as e:  # pragma: no cover - defensive
            print(f"[WARN] Could not remove {candidate}: {e}")


def aggregate_metric_tables(
    dataset_root: Path,
    args,
    excluded: set[tuple[str, str, str, str]] | None = None,
) -> dict[str, int]:
    """Rebuild the combined ERP/TFR/time-series tables from per-subject files."""
    specs = (
        (
            "erp_metrics",
            ERP_METRICS_PATTERN,
            dict(suffix="metrics", extension=".tsv", desc="erp"),
            "Dataset-level ERP metrics aggregated across processed subjects.",
        ),
        (
            "erp_timeseries",
            ERP_TIMESERIES_PATTERN,
            dict(suffix="timeseries", extension=".parquet", desc="erp"),
            "Dataset-level ERP time series aggregated across processed subjects.",
        ),
        (
            "tfr_metrics",
            TFR_METRICS_PATTERN,
            dict(suffix="metrics", extension=".tsv", desc="tfr"),
            "Dataset-level TFR metrics aggregated across processed subjects.",
        ),
    )

    counts: dict[str, int] = {}
    for name, pattern, path_kwargs, description in specs:
        df, n_files = _concat_subject_tables(dataset_root, pattern, excluded)
        counts[name] = n_files
        out_path = dataset_derivative_path(dataset_root, **path_kwargs)
        if df is None:
            # Nothing to build it from this gather: drop the previous version
            # rather than leave it looking current.
            _remove_dataset_output(out_path)
            continue
        _save_dataframe_with_sidecar(
            df,
            out_path,
            args,
            None,
            behavior_source=None,
            description=description,
        )
    return counts


def aggregate_grand_averages(
    dataset_root: Path,
    excluded: set[tuple[str, str, str, str]] | None = None,
) -> int:
    """Rebuild grand averages from per-subject evoked files.

    Evokeds are grouped by ``(session, task)`` and condition, matching how the
    serial path grouped its in-memory evokeds.
    """
    groups: dict[tuple[str | None, str | None], dict[str, list]] = {}
    # Conditions are keyed case-insensitively. The output path lower-cases the
    # condition, so grouping by raw label would let "Standard" (from a sidecar)
    # and "standard" (from a filename fallback) form two groups that then write
    # to the SAME path -- one grand average silently overwriting the other, each
    # built from only part of the cohort. Display casing is kept for metadata.
    display_labels: dict[str, str] = {}
    for path in _subject_scoped_files(dataset_root, EVOKED_PATTERN):
        if _is_excluded(path, excluded):
            continue
        condition = _condition_for_evoked(path)
        if not condition:
            continue
        try:
            entities = parse_bids_entities_like_name(path.stem)
            evoked = mne.read_evokeds(path, verbose="error")[0]
        except Exception as e:
            print(f"[WARN] Could not read evoked {path}: {e}")
            continue
        group_key = (entities.get("ses"), entities.get("task"))
        cond_key = condition.lower()
        display_labels.setdefault(cond_key, condition)
        groups.setdefault(group_key, {}).setdefault(cond_key, []).append(evoked)

    # Track what this gather actually produced so grand averages left by a
    # previous run -- a condition that no longer appears, or whose evokeds are now
    # all excluded or unreadable -- do not survive as though they were current.
    written_paths: set[Path] = set()
    n_written = 0
    for (ses, task), evoked_map in groups.items():
        group_entities: dict[str, str] = {}
        if ses:
            group_entities["ses"] = ses
        if task:
            group_entities["task"] = task
        for cond_key, ga in grand_averages(evoked_map).items():
            cond = display_labels.get(cond_key, cond_key)
            ga_path = dataset_derivative_path(
                dataset_root,
                entities=group_entities,
                suffix="ave",
                extension=".fif",
                desc=f"grandaverage-{cond_key}",
            )
            ga.save(ga_path, overwrite=True)
            write_json(
                derivative_sidecar_path(ga_path),
                {
                    "Description": f"Grand-average evoked response for {cond}.",
                    "GeneratedBy": [{"Name": PIPELINE_NAME, "Version": __version__}],
                    "Session": ses,
                    "Task": task,
                    "Condition": cond,
                },
            )
            written_paths.add(ga_path)
            n_written += 1

    for stale in _dataset_scoped_files(dataset_root, "*_ave.fif"):
        if stale not in written_paths:
            _remove_dataset_output(stale)

    return n_written


def run_aggregation(dataset_root: Path, args) -> dict[str, int]:
    """Rebuild every dataset-level output from per-subject derivatives.

    Safe to re-run: each output is overwritten from whatever per-subject files
    are currently on disk, so a gather job can be retried after a partial array
    run without duplicating rows.
    """
    dataset_root = Path(dataset_root)
    # Recordings the latest run skipped must not be resurrected from files an
    # earlier run left behind, or the QC summary would disagree with the tables.
    excluded = _excluded_recording_keys(dataset_root)
    counts = {"qc": aggregate_qc_summary(dataset_root)}
    counts.update(aggregate_metric_tables(dataset_root, args, excluded))
    counts["grand_averages"] = aggregate_grand_averages(dataset_root, excluded)
    if excluded:
        print(f"Excluded {len(excluded)} recording(s) from metrics/grand averages (QC status not OK).")

    if not counts["qc"]:
        print(f"[WARN] No per-subject QC rows found under {dataset_root}; nothing to aggregate.")
    else:
        print(
            "Aggregated {qc} QC row file(s), {erp_metrics} ERP / {tfr_metrics} TFR / "
            "{erp_timeseries} time-series table(s), {grand_averages} grand average(s).".format(**counts)
        )
    return counts


def run_aggregate_only(args) -> dict[str, int]:
    """``--aggregate_only`` entry point: gather without reprocessing anything."""
    from .cli_common import _finalize_runtime_paths, _prepare_derivatives_root

    _finalize_runtime_paths(args)
    dataset_root = _prepare_derivatives_root(args)
    counts = run_aggregation(dataset_root, args)
    print(f"Saved derivatives -> {dataset_root}")
    return counts
