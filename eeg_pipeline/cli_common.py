# ruff: noqa: F401
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from . import __version__
from .align import align_marker_positions_to_codes, keep_by_gap_heuristic, marker_gap_stats
from .artifacts import (
    moving_window_ptp_mask,
    moving_window_ptp_max,
    simple_voltage_threshold_mask,
    step_threshold_mask,
)
from .behavior import load_behavioral_events
from .bids import (
    PIPELINE_NAME,
    dataset_derivative_path,
    derivative_sidecar_path,
    ensure_derivatives_dataset,
    parse_bids_entities_like_name,
    source_basename_from_derivative_path,
    subject_derivative_path,
    write_json,
)
from .config import load_config
from .epoching import (
    EpochParams,
    build_events_from_positions_and_codes,
    make_epochs,
    select_and_filter_conditions,
    select_and_recode_stddev,
)
from .evoked import compute_evokeds, grand_averages
from .gpu import capability_report, format_capability_report
from .gpu import configure as configure_gpu
from .ica import ICAParams, apply_ica, find_ica_excludes, fit_ica
from .ica_diagnostics import compute_ica_diagnostics, recommend_ica
from .inputs import (
    PipelineRecording,
    convert_legacy_recordings_to_bids,
    discover_pipeline_recordings,
    subject_number_from_stem,
)
from .io_brainvision import (
    events_from_annotations_positions,
    parse_vmrk_markers,
    read_raw_preprocess,
)
from .metrics import compute_erp_metrics, compute_tfr_metrics, load_epochs
from .metrics.erp import ERPWindow
from .metrics.erp_timeseries import ERPTimeSeriesParams, compute_erp_timeseries
from .metrics.erp_windows import ERP_WINDOWS
from .metrics.tfr import TFRParams
from .qc import write_qc_summary
from .schema import parse_token_map


# Helper for config integration.  When merging configuration values
# into command‑line arguments we want to honour user‑supplied flags.
# This helper sets ``args.<field>`` only if the attribute still has
# its argparse default value.  The ``defaults`` dict is built once in
# ``main`` and passed through to ``run_full_pipeline``.  See
# ``build_defaults`` below for how defaults are collected.
def set_if_default(args, defaults: dict, field: str, value):
    """Assign a config value to ``args.field`` only if the CLI
    argument was not explicitly provided.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments namespace.  The field is looked up on this
        object and replaced if it still equals the parser default.
    defaults : dict
        Mapping of argument names to their argparse defaults.  This
        comes from ``build_defaults`` which iterates over all parser
        actions.
    field : str
        Name of the attribute on ``args`` to inspect and potentially
        overwrite.
    value : Any
        The value from the configuration that should be applied if
        appropriate.
    """
    # If the field wasn't in defaults we can't know the default so bail
    if field not in defaults:
        return
    # Only set the value if the user didn't override it via CLI
    if getattr(args, field) == defaults[field]:
        setattr(args, field, value)

_BV_KEY_RE = re.compile(r"^(?P<key>\w+)\s*=\s*(?P<val>.+?)\s*$", re.MULTILINE)

def _bv_get(txt: str, key: str) -> str | None:
    key_l = key.lower()
    for m in _BV_KEY_RE.finditer(txt):
        if m.group("key").strip().lower() == key_l:
            return m.group("val").strip()
    return None

def brainvision_links_ok(vhdr_path: Path) -> tuple[bool, str]:
    """
    Returns (ok, reason). Checks whether .vhdr's MarkerFile/DataFile exist.
    """
    txt = vhdr_path.read_text(encoding="utf-8", errors="replace")
    marker = _bv_get(txt, "MarkerFile")
    data = _bv_get(txt, "DataFile")

    missing = []
    if marker:
        if not (vhdr_path.parent / marker).exists():
            missing.append(f"MarkerFile={marker}")
    if data:
        if not (vhdr_path.parent / data).exists():
            missing.append(f"DataFile={data}")

    if missing:
        return False, "Missing referenced file(s): " + ", ".join(missing)
    return True, ""


def _pipeline_dataset_root(args) -> Path:
    return Path(args.derivatives_root) / PIPELINE_NAME


def _normalize_entity_filter_value(value: str | None, prefix: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith(f"{prefix}-"):
        return text[len(prefix) + 1 :]
    return text


def _input_mode(args, cfg: dict[str, Any] | None = None) -> str:
    if getattr(args, "legacy", False):
        return "legacy"
    mode = getattr(args, "input_mode", None)
    if mode not in (None, ""):
        return str(mode).lower()
    if cfg is not None:
        return str(cfg.get("input", {}).get("mode", "bids")).lower()
    return "bids"


def _coerce_path_arg(args, field: str) -> Path | None:
    value = getattr(args, field, None)
    if value in (None, ""):
        setattr(args, field, None)
        return None
    if isinstance(value, Path):
        return value
    path = Path(value)
    setattr(args, field, path)
    return path


def _legacy_task_label(args, cfg: dict[str, Any] | None = None) -> str | None:
    explicit = getattr(args, "task_label", None)
    if explicit not in (None, ""):
        return _normalize_entity_filter_value(str(explicit), "task")
    tasks = getattr(args, "tasks", None)
    if isinstance(tasks, (list, tuple)) and len(tasks) == 1:
        return _normalize_entity_filter_value(str(tasks[0]), "task")
    if cfg is not None:
        task_value = cfg.get("task", None)
        if task_value not in (None, ""):
            return str(task_value)
    return None


def _finalize_runtime_paths(args, cfg: dict[str, Any] | None = None) -> None:
    args.input_mode = _input_mode(args, cfg)

    for field in (
        "bids_root",
        "raw_dir",
        "subject_csv_dir",
        "derivatives_root",
        "sourcedata_root",
        "behavior_csv_fallback_dir",
        "conversion_bids_root",
    ):
        _coerce_path_arg(args, field)

    if bool(getattr(args, "convert_to_bids", False)) and getattr(args, "conversion_bids_root", None) is None:
        raw_dir = getattr(args, "raw_dir", None)
        if raw_dir is not None:
            args.conversion_bids_root = raw_dir.parent / f"{raw_dir.name}_bids"

    if getattr(args, "derivatives_root", None) is None:
        if args.input_mode == "bids" and getattr(args, "bids_root", None) is not None:
            args.derivatives_root = Path(args.bids_root) / "derivatives"
        elif (
            args.input_mode == "legacy"
            and bool(getattr(args, "convert_to_bids", False))
            and getattr(args, "conversion_bids_root", None) is not None
        ):
            args.derivatives_root = Path(args.conversion_bids_root) / "derivatives"
        elif args.input_mode == "legacy" and getattr(args, "raw_dir", None) is not None:
            args.derivatives_root = Path(args.raw_dir).parent / "derivatives"


def _expected_behavior_events_path(recording: PipelineRecording) -> Path:
    if recording.behavior_kind == "bids_events" and recording.behavior_path is not None:
        return recording.behavior_path
    return recording.raw_path.with_name(f"{recording.raw_path.stem}_events.tsv")


def _behavior_inputs_for_recording(
    recording: PipelineRecording,
) -> tuple[Path, Path, Path | None, Path]:
    if recording.behavior_kind == "bids_events" and recording.behavior_path is not None:
        events_tsv = recording.behavior_path
        events_json = recording.behavior_json_path or recording.behavior_path.with_suffix(".json")
        return events_tsv, events_json, None, recording.behavior_path

    csv_path = None
    if recording.behavior_kind == "csv" and recording.behavior_path is not None and recording.behavior_path.exists():
        csv_path = recording.behavior_path
    events_tsv = _expected_behavior_events_path(recording)
    return events_tsv, events_tsv.with_suffix(".json"), csv_path, (recording.behavior_path or events_tsv)


def _recording_from_raw_path(
    args,
    raw_path: Path,
    cfg: dict[str, Any] | None = None,
) -> PipelineRecording:
    mode = _input_mode(args, cfg)
    task_label = _legacy_task_label(args, cfg)
    source_root = raw_path.parent if mode == "legacy" else _infer_bids_root(raw_path)
    recordings = discover_pipeline_recordings(
        mode=mode,
        bids_root=source_root if mode == "bids" else None,
        raw_dir=source_root if mode == "legacy" else None,
        subject_csv_dir=getattr(args, "subject_csv_dir", None),
        subjects=None,
        sessions=None,
        tasks=None,
        runs=None,
        task_label=task_label,
    )
    for recording in recordings:
        if recording.raw_path == raw_path:
            return recording

    try:
        entities = parse_bids_entities_like_name(raw_path.stem)
    except ValueError:
        entities = {}
    entities.setdefault("sub", subject_number_from_stem(raw_path.stem))
    if task_label and "task" not in entities:
        entities["task"] = task_label
    behavior_path = None
    behavior_kind = "none"
    if mode == "bids":
        behavior_path = raw_path.with_name(f"{raw_path.stem.replace('_eeg', '')}_events.tsv")
        behavior_kind = "bids_events"
    return PipelineRecording(
        source_type=mode,
        source_root=source_root,
        raw_path=raw_path,
        entities=entities,
        behavior_path=behavior_path,
        behavior_json_path=None if behavior_path is None else behavior_path.with_suffix(".json"),
        behavior_kind=behavior_kind,
    )


def _prepare_derivatives_root(args, *, source_dataset: Path | None = None) -> Path:
    if source_dataset is None:
        if getattr(args, "bids_root", None) not in (None, ""):
            source_dataset = Path(args.bids_root)
        elif getattr(args, "raw_dir", None) not in (None, ""):
            source_dataset = Path(args.raw_dir)
    return ensure_derivatives_dataset(
        Path(args.derivatives_root),
        source_dataset=source_dataset,
        pipeline_version=__version__,
    )


def _dataset_metrics_dir(dataset_root: Path) -> Path:
    metrics_dir = dataset_root / "eeg"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir


def _subject_metrics_dir(dataset_root: Path, recording: PipelineRecording) -> Path:
    return subject_derivative_path(
        dataset_root,
        recording.entities,
        suffix="epo",
        extension=".fif",
    ).parent


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _normalize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(v) for v in value]
    return value


def _processing_metadata(
    args,
    recording: PipelineRecording,
    *,
    behavior_source: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        behavior_ref = str(behavior_source.relative_to(recording.source_root))
    except ValueError:
        behavior_ref = str(behavior_source)
    metadata: dict[str, Any] = {
        "Description": "Generated by eeg-pipeline from an EEG recording.",
        "Sources": [
            recording.relative_raw_path,
            behavior_ref,
        ],
        "GeneratedBy": [{"Name": PIPELINE_NAME, "Version": __version__}],
        "Preprocessing": {
            "Montage": args.montage,
            "Rereference": args.reref,
            "HighPassHz": float(args.l_freq),
            "LowPassHz": float(args.h_freq),
            "NotchHz": list(args.notch) if args.notch else [],
        },
        "Epoching": {
            "Tmin": float(args.tmin),
            "Tmax": float(args.tmax),
            "Baseline": [float(args.baseline[0]), float(args.baseline[1])],
        },
        "ArtifactRejection": {
            "TestWindow": [float(args.art_test_tmin), float(args.art_test_tmax)],
            "Blink": {
                "ThresholdUV": float(args.blink_threshold_uv),
                "WindowMs": float(args.blink_win_ms),
                "StepMs": float(args.blink_step_ms),
                "AutoPercentile": getattr(args, "blink_auto_percentile", None),
            },
            "Voltage": {
                "Method": getattr(args, "volt_method", "simple"),
                "PositiveThresholdUV": float(args.volt_pos_uv),
                "NegativeThresholdUV": float(args.volt_neg_uv),
                "PTPThresholdUV": float(getattr(args, "volt_threshold_uv", 150.0)),
                "WindowMs": float(getattr(args, "volt_win_ms", 200.0)),
                "StepMs": float(getattr(args, "volt_step_ms", 10.0)),
                "StepThresholdUVPerMs": getattr(args, "volt_step_uv_per_ms", None),
                "AutoPercentile": getattr(args, "volt_auto_percentile", None),
                "MaxRejectRate": getattr(args, "max_reject_rate", None),
            },
        },
        "ICA": {
            "Mode": args.ica,
            "Method": getattr(args, "ica_method", None),
            "NComponents": _parse_n_components(getattr(args, "ica_n_components", None)),
            "RandomState": getattr(args, "ica_random_state", None),
            "MaxIter": getattr(args, "ica_max_iter", None),
            "FitHighPassHz": getattr(args, "ica_fit_l_freq", None),
            "FitLowPassHz": getattr(args, "ica_fit_h_freq", None),
            "Decim": getattr(args, "ica_decim", None),
            "CorrThreshold": getattr(args, "ica_corr_thresh", None),
            "MaxExclude": getattr(args, "ica_max_exclude", None),
        },
    }
    if extra:
        metadata.update(extra)
    return _normalize_json_value(metadata)


def _write_output_sidecar(
    data_path: Path,
    args,
    recording: PipelineRecording,
    *,
    behavior_source: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    write_json(
        derivative_sidecar_path(data_path),
        _processing_metadata(args, recording, behavior_source=behavior_source, extra=extra),
    )


def _save_dataframe_with_sidecar(
    df: pd.DataFrame,
    data_path: Path,
    args,
    recording: PipelineRecording | None,
    *,
    behavior_source: Path | None,
    description: str,
    column_descriptions: dict[str, Any] | None = None,
) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.suffix == ".parquet":
        df.to_parquet(data_path, index=False)
    else:
        sep = "\t" if data_path.suffix == ".tsv" else ","
        df.to_csv(data_path, sep=sep, index=False)

    metadata: dict[str, Any] = {"Description": description}
    if column_descriptions:
        metadata.update(column_descriptions)
    if recording is not None and behavior_source is not None:
        _write_output_sidecar(
            data_path,
            args,
            recording,
            behavior_source=behavior_source,
            extra=metadata,
        )
    else:
        write_json(derivative_sidecar_path(data_path), _normalize_json_value(metadata))


def _events_json_sidecar(events_df: pd.DataFrame, trial_type_levels: dict[str, str] | None = None) -> dict[str, Any]:
    sidecar: dict[str, Any] = {
        "onset": {
            "Description": "Event onset in seconds relative to the start of the recording.",
        },
        "duration": {
            "Description": "Event duration in seconds.",
        },
        "sample": {
            "Description": "Integer sample index used for epoching and derivative event export.",
        },
        "value": {
            "Description": "Numeric event code used by eeg-pipeline for filtering and alignment.",
        },
        "source_event_index": {
            "Description": "Row index from the source events table before alignment or filtering.",
        },
    }
    if "trial_type" in events_df.columns:
        trial_type = {
            "Description": "Condition label carried forward into the derivative events export.",
        }
        if trial_type_levels:
            trial_type["Levels"] = trial_type_levels
        sidecar["trial_type"] = trial_type
    return sidecar


def _finalized_events_table(
    metadata: pd.DataFrame,
    *,
    sfreq: float,
    samples: np.ndarray,
    codes: np.ndarray,
) -> pd.DataFrame:
    finalized = metadata.copy().reset_index(drop=True)
    finalized["source_event_index"] = np.arange(len(finalized), dtype=int)
    finalized["sample"] = np.asarray(samples, dtype=int)
    finalized["onset"] = finalized["sample"] / float(sfreq)
    if "duration" not in finalized.columns:
        finalized["duration"] = 0.0
    finalized["value"] = np.asarray(codes, dtype=int)

    base_columns = ["onset", "duration", "sample"]
    optional_columns = [c for c in ("trial_type", "value", "code", "condition") if c in finalized.columns]
    remaining = [c for c in finalized.columns if c not in set(base_columns + optional_columns)]
    return finalized[base_columns + optional_columns + remaining]


def _group_key(recording: PipelineRecording, condition: str) -> tuple[str | None, str | None, str]:
    return recording.session_id, recording.task_id, condition


def _infer_bids_root(raw_path: Path) -> Path:
    for candidate in [raw_path.parent, *raw_path.parents]:
        if (candidate / "dataset_description.json").exists():
            return candidate
    for parent in raw_path.parents:
        if parent.name.startswith("sub-"):
            return parent.parent
    raise ValueError(f"Could not infer BIDS root for {raw_path}")

def _parse_n_components(x):
    """
    MNE ICA n_components can be float (variance fraction) or int (#components).
    argparse gives us a string; infer int vs float.
    """
    if x is None:
        return 0.99
    if isinstance(x, (int, float)):
        return x
    s = str(x).strip()
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return float(s)




def detect_trigger_bursts(markers_pos: np.ndarray, sfreq: float,
                          min_iti_s: float = 0.02,
                          burst_win_s: float = 0.25,
                          burst_count: int = 5) -> dict:
    """
    Detect suspicious StimTrak behavior:
      - very short ITIs (<= min_iti_s)
      - bursts: >= burst_count triggers inside burst_win_s

    Returns summary diagnostics; does NOT modify markers.
    """
    if len(markers_pos) < 2:
        return {
            "burst_flag": False,
            "n_triggers": int(len(markers_pos)),
            "n_short_iti": 0,
            "min_iti_s": None,
            "burst_max_in_window": 1,
            "burst_n_windows_ge_thresh": 0,
        }

    t = markers_pos / float(sfreq)
    dt = np.diff(t)

    n_short = int(np.sum(dt <= min_iti_s))
    min_iti = float(np.min(dt))

    # Sliding window burst count using two pointers
    j = 0
    burst_max = 1
    n_ge = 0
    for i in range(len(t)):
        while t[i] - t[j] > burst_win_s:
            j += 1
        c = i - j + 1
        burst_max = max(burst_max, c)
        if c >= burst_count:
            n_ge += 1

    burst_flag = (n_short > 0) or (burst_max >= burst_count)

    return {
        "burst_flag": bool(burst_flag),
        "n_triggers": int(len(markers_pos)),
        "n_short_iti": int(n_short),
        "min_iti_s": min_iti,
        "burst_max_in_window": int(burst_max),
        "burst_n_windows_ge_thresh": int(n_ge),
        "burst_params": f"min_iti_s={min_iti_s},win_s={burst_win_s},count={burst_count}",
    }


