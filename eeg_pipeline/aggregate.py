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
from pathlib import Path

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


def _concat_subject_tables(dataset_root: Path, pattern: str) -> tuple[pd.DataFrame | None, int]:
    """Concatenate every per-subject table matching ``pattern``.

    Returns the combined frame (or ``None``) and the number of files contributing
    to it, so callers can report coverage instead of silently emitting a table
    built from fewer subjects than expected.
    """
    frames: list[pd.DataFrame] = []
    for path in _subject_scoped_files(dataset_root, pattern):
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


def aggregate_metric_tables(dataset_root: Path, args) -> dict[str, int]:
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
        df, n_files = _concat_subject_tables(dataset_root, pattern)
        counts[name] = n_files
        if df is None:
            continue
        _save_dataframe_with_sidecar(
            df,
            dataset_derivative_path(dataset_root, **path_kwargs),
            args,
            None,
            behavior_source=None,
            description=description,
        )
    return counts


def aggregate_grand_averages(dataset_root: Path) -> int:
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
            n_written += 1
    return n_written


def run_aggregation(dataset_root: Path, args) -> dict[str, int]:
    """Rebuild every dataset-level output from per-subject derivatives.

    Safe to re-run: each output is overwritten from whatever per-subject files
    are currently on disk, so a gather job can be retried after a partial array
    run without duplicating rows.
    """
    dataset_root = Path(dataset_root)
    counts = {"qc": aggregate_qc_summary(dataset_root)}
    counts.update(aggregate_metric_tables(dataset_root, args))
    counts["grand_averages"] = aggregate_grand_averages(dataset_root)

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
