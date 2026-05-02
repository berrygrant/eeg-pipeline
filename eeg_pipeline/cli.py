# mmn_pipeline/cli.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import mne

from . import __version__
from .schema import parse_token_map
from .behavior import load_behavioral_events
from .io_brainvision import read_raw_preprocess, events_from_annotations_positions, parse_vmrk_markers
from .align import marker_gap_stats, keep_by_gap_heuristic, align_marker_positions_to_codes
from .epoching import (
    EpochParams,
    build_events_from_positions_and_codes,
    select_and_recode_stddev,
    select_and_filter_conditions,
    make_epochs,
)
from .artifacts import (
    moving_window_ptp_mask,
    moving_window_ptp_max,
    simple_voltage_threshold_mask,
    step_threshold_mask,
)
from .evoked import compute_evokeds, grand_averages
from .qc import write_qc_summary
from .ica_diagnostics import compute_ica_diagnostics, recommend_ica
from .ica import ICAParams, fit_ica, find_ica_excludes, apply_ica
from .gpu import configure as configure_gpu, capability_report, format_capability_report
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
from .inputs import (
    PipelineRecording,
    convert_legacy_recordings_to_bids,
    discover_pipeline_recordings,
    subject_number_from_stem,
)

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
from eeg_pipeline.config import load_config

# Metrics (ERP + TFR)
from eeg_pipeline.metrics import compute_erp_metrics, compute_tfr_metrics, load_epochs
from eeg_pipeline.metrics.erp import ERPWindow
from eeg_pipeline.metrics.erp_windows import ERP_WINDOWS
from eeg_pipeline.metrics.erp_timeseries import ERPTimeSeriesParams, compute_erp_timeseries
from eeg_pipeline.metrics.tfr import TFRParams

import re
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


def summarize_one_file(args, raw_path: Path):
    _finalize_runtime_paths(args)
    recording = _recording_from_raw_path(args, raw_path)
    subj = recording.subject_label
    is_bv = raw_path.suffix.lower() == ".vhdr"
    vmrk_path = raw_path.with_suffix(".vmrk") if is_bv else None

    print(f"\n=== SUMMARY: {subj} ===")
    print("Raw file:", raw_path)
    if recording.behavior_kind == "bids_events":
        print("BIDS events:", recording.behavior_path)
    elif recording.behavior_kind == "csv":
        print("Legacy behavior CSV:", recording.behavior_path)
    else:
        print("Behavior source:", recording.behavior_path)
    if is_bv:
        print("VMRK file:", vmrk_path)

    # Show annotation descriptions without any preprocessing (debug)
    if is_bv:
        raw0 = mne.io.read_raw_brainvision(raw_path, preload=True)
    else:
        raw0 = mne.io.read_raw_eeglab(raw_path, preload=True)
    descs = list(dict.fromkeys(raw0.annotations.description))
    print("\nAnnotation descriptions (first 30 unique):")
    print(descs[:30])
    print("Unique annotation count:", len(set(raw0.annotations.description)))

    # Preprocess (montage/reference/filter)
    raw = read_raw_preprocess(
        raw_path=raw_path,
        montage=args.montage,
        eog_chs=args.eog_chs,
        aux_chs=args.aux_chs,
        reref=args.reref,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        notch=args.notch,
    )

    # ICA diagnostics (non-destructive)
    ica_diag = compute_ica_diagnostics(
        raw,
        blink_proxy_chs=args.blink_proxy_chs,
        blink_threshold_uv=args.blink_threshold_uv,
        blink_win_ms=args.blink_win_ms,
        blink_step_ms=args.blink_step_ms,
    )
    print("\nICA diagnostics:")
    print(pd.Series(ica_diag).to_string())

    # If no true EOG rate, recommend_ica can use proxy
    pre_epoch_reco = recommend_ica(
        epoch_reject_rate=0.0,  # unknown yet
        eog_corr_max=ica_diag.get("eog_corr_max", 0.0),
        blink_rate_per_min=ica_diag.get("blink_rate_per_min", 0.0),
        blink_proxy_rate_per_min=ica_diag.get("blink_proxy_rate_per_min", 0.0),
        epoch_loss_thresh=0.20,                 # won’t trigger since 0.0
        eog_corr_thresh=args.ica_corr_thresh,
        blink_rate_thresh=args.ica_auto_blink_rate_per_min,
    )

    events_ann = events_from_annotations_positions(raw)
    markers_pos = events_ann[:, 0].copy()

    # ------------------------------------------------------------
    # StimTrak trigger QC: detect burst-like trigger failures
    # (do NOT modify markers; flag only)
    # ------------------------------------------------------------
    burst_diag = detect_trigger_bursts(
        markers_pos=markers_pos,
        sfreq=float(raw.info["sfreq"]),
        min_iti_s=0.02,      # 20 ms: impossible for real trials
        burst_win_s=0.25,    # 250 ms window
        burst_count=5,       # ≥5 triggers in 250 ms
    )

    trigger_diag = {
        "trigger_burst_flag": burst_diag["burst_flag"],
        "trigger_n_short_iti": burst_diag["n_short_iti"],
        "trigger_min_iti_s": burst_diag["min_iti_s"],
        "trigger_burst_max_in_window": burst_diag["burst_max_in_window"],
        "trigger_burst_n_windows_ge_thresh": burst_diag["burst_n_windows_ge_thresh"],
        "trigger_burst_params": burst_diag.get("burst_params", ""),
    }

    burst_qc = {
        "trigger_burst_flag": bool(burst_diag.get("burst_flag", False)),
        "trigger_n_short_iti": int(burst_diag.get("n_short_iti", 0) or 0),
        "trigger_min_iti_s": burst_diag.get("min_iti_s", ""),
        "trigger_burst_max_in_window": int(burst_diag.get("burst_max_in_window", 1) or 1),
        "trigger_burst_n_windows_ge_thresh": int(burst_diag.get("burst_n_windows_ge_thresh", 0) or 0),
        "trigger_burst_params": burst_diag.get("burst_params", ""),
    }

    if burst_diag["burst_flag"]:
        print(f"[WARN] Trigger burst detected for {subj}: "
              f"short_iti={burst_diag['n_short_iti']}, "
              f"max_in_window={burst_diag['burst_max_in_window']}")
        print("\nTotal events (from annotations):", len(events_ann))
        print("Event ID distribution (from annotations):")
        print(pd.Series(events_ann[:, 2]).value_counts().sort_index().to_string())

    stats = marker_gap_stats(markers_pos, sfreq=float(raw.info["sfreq"]))
    print("\nInter-marker gap stats (seconds):")
    for k in ["dt_min", "dt_p25", "dt_p50", "dt_p75", "dt_p90", "dt_p95", "dt_p99", "dt_max"]:
        if k in stats:
            print(f"  {k}: {stats[k]:.4f}")

    cand_gaps = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    print("\nKeep counts for candidate --drop_eeg_markers_by_gap_s values:")
    for g in cand_gaps:
        keep_idx = keep_by_gap_heuristic(markers_pos, sfreq=float(raw.info["sfreq"]), gap_s=g)
        print(f"  gap_s={g:>4}: keep {len(keep_idx)}/{len(markers_pos)}")

    # Parse .vmrk if present (debug)
    if is_bv and vmrk_path and vmrk_path.exists():
        mk = parse_vmrk_markers(vmrk_path)
        print("\nMarkers from .vmrk:")
        print("  total markers:", len(mk))
        if len(mk):
            print("  marker types:\n", mk["mtype"].value_counts().to_string())
            print("  unique desc count:", mk["desc"].nunique())
            print("  desc distribution (top 10):\n", mk["desc"].value_counts().head(10).to_string())
    elif is_bv:
        print("\n[WARN] .vmrk file not found next to .vhdr; cannot parse markers directly.")

    token_map = parse_token_map(args.token_map)
    csv_fallback_dir = (
        None
        if getattr(args, "behavior_csv_fallback_dir", None) in (None, "")
        else Path(args.behavior_csv_fallback_dir)
    )
    events_tsv, events_json, csv_path, _ = _behavior_inputs_for_recording(recording)
    try:
        behavior = load_behavioral_events(
            events_tsv=events_tsv,
            events_json=events_json,
            subject_id=recording.subject_id,
            keep_codes=args.behavioral_keep_codes,
            token_map=token_map,
            condition_map=getattr(args, "condition_map", None),
            csv_path=csv_path,
            csv_fallback_dir=csv_fallback_dir,
        )
    except FileNotFoundError as exc:
        print("\n[WARN]", str(exc))
        print("Cannot summarize behavioral events without source events or an explicit CSV fallback. Exiting summary.")
        return

    codes_all = behavior.codes_all
    print("\nBehavioral codes (EventCode) count:", len(codes_all))
    print("Behavioral code distribution:")
    print(pd.Series(codes_all).value_counts().sort_index().to_string())

    codes = behavior.codes
    if args.behavioral_keep_codes:
        print("\nBehavioral keep-codes filter applied:")
        print("  keep codes:", list(map(int, args.behavioral_keep_codes)))
        print("  remaining codes:", len(codes))

    print("\nSanity check (Step 4):")
    print("  EEG markers available:", len(markers_pos))
    print("  behavioral codes to assign:", len(codes))

    if behavior.samples is not None:
        aligned = behavior.samples
        diag = {
            "markers_original": int(len(markers_pos)),
            "markers_dropped_by_gap": 0,
            "markers_dropped_by_auto": 0,
        }
        print("  Using BIDS events.tsv sample column directly; EEG marker alignment skipped.")
    else:
        aligned, diag = align_marker_positions_to_codes(
            markers_pos=markers_pos,
            sfreq=float(raw.info["sfreq"]),
            codes=codes,
            gap_s=args.drop_eeg_markers_by_gap_s,
            auto_drop_to_count=bool(args.auto_drop_to_count),
        )
    print("  [OK] alignment achievable.")
    print(
        f"  Alignment: markers {diag['markers_original']} -> {len(aligned)} "
        f"(gap_drop={diag['markers_dropped_by_gap']}, auto_drop={diag['markers_dropped_by_auto']})"
    )

    print("\nToken map:", token_map)
    print("Metadata preview (first 5 rows):")
    print(behavior.metadata.head(5).to_string(index=False))

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


def apply_config(args, defaults=None):
    """Load config and apply values to args (respecting CLI overrides)."""
    if defaults is None:
        defaults = {}
    cfg = load_config(args.config)

    # Paths
    set_if_default(args, defaults, "raw_dir", cfg["paths"].get("raw_dir"))
    set_if_default(args, defaults, "subject_csv_dir", cfg["paths"].get("subject_csv_dir"))
    set_if_default(args, defaults, "bids_root", cfg["paths"]["bids_root"])
    set_if_default(args, defaults, "derivatives_root", cfg["paths"]["derivatives_root"])
    set_if_default(args, defaults, "sourcedata_root", cfg["paths"].get("sourcedata_root"))
    set_if_default(args, defaults, "conversion_bids_root", cfg.get("conversion", {}).get("bids_output_root"))
    set_if_default(
        args,
        defaults,
        "conversion_overwrite",
        int(bool(cfg.get("conversion", {}).get("overwrite", getattr(args, "conversion_overwrite", 1)))),
    )
    set_if_default(
        args,
        defaults,
        "convert_to_bids",
        bool(cfg.get("conversion", {}).get("enabled", getattr(args, "convert_to_bids", False))),
    )
    set_if_default(args, defaults, "task_label", cfg.get("task", getattr(args, "task_label", None)))
    if getattr(args, "legacy", False):
        args.input_mode = "legacy"
    else:
        args.input_mode = str(cfg.get("input", {}).get("mode", "bids")).lower()

    # BIDS discovery filters
    bids_cfg = cfg.get("bids", {})
    if getattr(args, "subjects", None) is None:
        args.subjects = bids_cfg.get("subjects", None)
    if getattr(args, "sessions", None) is None:
        args.sessions = bids_cfg.get("sessions", None)
    if getattr(args, "tasks", None) is None:
        args.tasks = bids_cfg.get("tasks", None)
    if getattr(args, "runs", None) is None:
        args.runs = bids_cfg.get("runs", None)

    # Channels and preprocessing
    set_if_default(args, defaults, "montage", cfg["preprocess"].get("montage", args.montage))
    set_if_default(args, defaults, "reref", cfg["preprocess"].get("reref", args.reref))
    set_if_default(args, defaults, "l_freq", cfg["preprocess"].get("l_freq", args.l_freq))
    set_if_default(args, defaults, "h_freq", cfg["preprocess"].get("h_freq", args.h_freq))
    set_if_default(args, defaults, "notch", cfg["preprocess"].get("notch_hz", args.notch))

    # Channel selections
    set_if_default(args, defaults, "eog_chs", cfg["channels"].get("eog_chs", args.eog_chs))
    set_if_default(args, defaults, "blink_proxy_chs", cfg["channels"].get("blink_proxy_chs", args.blink_proxy_chs))
    set_if_default(args, defaults, "aux_chs", cfg["channels"].get("drop_aux_chs", args.aux_chs))

    # Events
    set_if_default(args, defaults, "standard_codes", cfg["events"].get("standard_codes", args.standard_codes))
    set_if_default(args, defaults, "deviant_codes", cfg["events"].get("deviant_codes", args.deviant_codes))
    set_if_default(
        args, defaults, "behavioral_keep_codes",
        cfg["events"].get("behavioral_keep_codes", args.behavioral_keep_codes)
    )
    set_if_default(
        args, defaults, "drop_eeg_markers_by_gap_s",
        cfg["events"].get("drop_eeg_markers_by_gap_s", args.drop_eeg_markers_by_gap_s)
    )
    set_if_default(
        args, defaults, "behavior_csv_fallback_dir",
        cfg["events"].get("csv_fallback_dir", getattr(args, "behavior_csv_fallback_dir", None)),
    )
    set_if_default(
        args, defaults, "auto_drop_to_count",
        int(bool(cfg["events"].get("auto_drop_to_count", args.auto_drop_to_count)))
    )
    # Optional condition map (name -> code)
    cond_map = cfg["events"].get("condition_map", None)
    if cond_map is not None:
        setattr(args, "condition_map", cond_map)

    # Epoching
    set_if_default(args, defaults, "tmin", cfg["epoching"].get("tmin", args.tmin))
    set_if_default(args, defaults, "tmax", cfg["epoching"].get("tmax", args.tmax))
    set_if_default(args, defaults, "baseline", cfg["epoching"].get("baseline", args.baseline))

    # Artifacts
    art = cfg.get("artifacts", {})
    win = art.get("test_window", [args.art_test_tmin, args.art_test_tmax])
    if len(win) >= 2:
        set_if_default(args, defaults, "art_test_tmin", float(win[0]))
        set_if_default(args, defaults, "art_test_tmax", float(win[1]))
    blink_cfg = art.get("blink", {})
    set_if_default(args, defaults, "blink_threshold_uv", blink_cfg.get("threshold_uv", args.blink_threshold_uv))
    set_if_default(args, defaults, "blink_win_ms", blink_cfg.get("win_ms", args.blink_win_ms))
    set_if_default(args, defaults, "blink_step_ms", blink_cfg.get("step_ms", args.blink_step_ms))
    set_if_default(args, defaults, "blink_auto_percentile", blink_cfg.get("auto_percentile", args.blink_auto_percentile))
    volt_cfg = art.get("voltage", {})
    set_if_default(args, defaults, "volt_pos_uv", volt_cfg.get("pos_uv", args.volt_pos_uv))
    set_if_default(args, defaults, "volt_neg_uv", volt_cfg.get("neg_uv", args.volt_neg_uv))
    # Optional windowed EEG artifact rejection (if configured)
    if "volt_method" not in defaults:
        setattr(args, "volt_method", volt_cfg.get("method", "simple"))
    else:
        set_if_default(args, defaults, "volt_method", volt_cfg.get("method", args.volt_method))
    if "volt_threshold_uv" not in defaults:
        setattr(args, "volt_threshold_uv", volt_cfg.get("threshold_uv", 150.0))
    else:
        set_if_default(args, defaults, "volt_threshold_uv", volt_cfg.get("threshold_uv", args.volt_threshold_uv))
    if "volt_win_ms" not in defaults:
        setattr(args, "volt_win_ms", volt_cfg.get("win_ms", 200.0))
    else:
        set_if_default(args, defaults, "volt_win_ms", volt_cfg.get("win_ms", args.volt_win_ms))
    if "volt_step_ms" not in defaults:
        setattr(args, "volt_step_ms", volt_cfg.get("step_ms", 10.0))
    else:
        set_if_default(args, defaults, "volt_step_ms", volt_cfg.get("step_ms", args.volt_step_ms))
    if "volt_step_uv_per_ms" not in defaults:
        setattr(args, "volt_step_uv_per_ms", volt_cfg.get("step_uv_per_ms", None))
    else:
        set_if_default(args, defaults, "volt_step_uv_per_ms", volt_cfg.get("step_uv_per_ms", args.volt_step_uv_per_ms))
    if "volt_auto_percentile" not in defaults:
        setattr(args, "volt_auto_percentile", volt_cfg.get("auto_percentile", None))
    else:
        set_if_default(args, defaults, "volt_auto_percentile", volt_cfg.get("auto_percentile", args.volt_auto_percentile))

    if "max_reject_rate" not in defaults:
        setattr(args, "max_reject_rate", art.get("max_reject_rate", None))
    else:
        set_if_default(args, defaults, "max_reject_rate", art.get("max_reject_rate", args.max_reject_rate))

    # ICA
    ica_cfg = cfg.get("ica", {})
    set_if_default(args, defaults, "ica", ica_cfg.get("mode", args.ica))
    set_if_default(
        args, defaults, "ica_auto_blink_rate_per_min",
        ica_cfg.get("auto_blink_rate_per_min", args.ica_auto_blink_rate_per_min)
    )
    set_if_default(args, defaults, "ica_method", ica_cfg.get("method", args.ica_method))
    set_if_default(args, defaults, "ica_n_components", str(ica_cfg.get("n_components", args.ica_n_components)))
    set_if_default(args, defaults, "ica_random_state", ica_cfg.get("random_state", args.ica_random_state))
    set_if_default(args, defaults, "ica_max_iter", ica_cfg.get("max_iter", args.ica_max_iter))
    set_if_default(args, defaults, "ica_fit_l_freq", ica_cfg.get("fit_l_freq", args.ica_fit_l_freq))
    set_if_default(args, defaults, "ica_fit_h_freq", ica_cfg.get("fit_h_freq", args.ica_fit_h_freq))
    set_if_default(args, defaults, "ica_decim", ica_cfg.get("decim", args.ica_decim))
    set_if_default(args, defaults, "ica_corr_thresh", ica_cfg.get("corr_thresh", args.ica_corr_thresh))
    set_if_default(args, defaults, "ica_max_exclude", ica_cfg.get("max_exclude", args.ica_max_exclude))
    set_if_default(args, defaults, "save_ica", int(bool(ica_cfg.get("save_ica", args.save_ica))))

    # Metrics
    metrics_cfg = cfg.get("metrics", {})
    erp_cfg = metrics_cfg.get("erp", {}) if isinstance(metrics_cfg.get("erp", {}), dict) else {}
    tfr_cfg = metrics_cfg.get("tfr", {}) if isinstance(metrics_cfg.get("tfr", {}), dict) else {}

    erp_enabled = bool(erp_cfg.get("enabled", True))
    tfr_enabled = bool(tfr_cfg.get("enabled", False))

    if "enabled" in metrics_cfg:
        metrics_enabled = bool(metrics_cfg.get("enabled"))
    else:
        metrics_enabled = bool(erp_enabled or tfr_enabled)
    set_if_default(args, defaults, "metrics", int(metrics_enabled))

    # Stash per‑modality enable flags for later gating (no CLI flags)
    args.metrics_erp_enabled = erp_enabled
    args.metrics_tfr_enabled = tfr_enabled
    args.metrics_erp_timeseries = bool(erp_cfg.get("timeseries", False))

    # Only override these from config when the user didn't specify them
    if args.metrics_channels is None:
        chs = erp_cfg.get("channels", None)
        if chs is None:
            chs = metrics_cfg.get("channels", None)
        if isinstance(chs, (list, tuple)) and len(chs):
            args.metrics_channels = list(map(str, chs))

    # Optional metrics conditions (for ERP/TFR)
    if getattr(args, "metrics_conditions", None) is None:
        conds = erp_cfg.get("conditions", None)
        if conds is None:
            conds = metrics_cfg.get("conditions", None)
        if isinstance(conds, (list, tuple)) and len(conds):
            args.metrics_conditions = list(map(str, conds))
        elif conds is not None:
            args.metrics_conditions = [str(conds)]

    # ERP windows: list[dict] (preferred) or list[list/tuple]
    if args.erp_window is None:
        wins = erp_cfg.get("windows", None)
        if wins is None:
            wins = metrics_cfg.get("erp_windows", None)
        if isinstance(wins, list) and len(wins):
            parsed = []
            for w in wins:
                if isinstance(w, dict):
                    name = str(w.get("name", "window"))
                    tmin = float(w.get("tmin"))
                    tmax = float(w.get("tmax"))
                    parsed.append([name, tmin, tmax])
                elif isinstance(w, (list, tuple)) and len(w) >= 3:
                    parsed.append([str(w[0]), float(w[1]), float(w[2])])
            if parsed:
                args.erp_window = parsed

    set_if_default(
        args,
        defaults,
        "compute_mmn",
        int(bool(erp_cfg.get("compute_mmn", metrics_cfg.get("compute_mmn", args.compute_mmn)))),
    )
    set_if_default(
        args,
        defaults,
        "difference_label",
        erp_cfg.get("difference_label", metrics_cfg.get("difference_label", args.difference_label)),
    )
    set_if_default(
        args,
        defaults,
        "compute_p300",
        int(bool(erp_cfg.get("compute_p300", metrics_cfg.get("compute_p300", args.compute_p300)))),
    )

    set_if_default(args, defaults, "tfr_tmin", float(tfr_cfg.get("tmin", args.tfr_tmin)))
    set_if_default(args, defaults, "tfr_tmax", float(tfr_cfg.get("tmax", args.tfr_tmax)))
    set_if_default(args, defaults, "tfr_fmin", float(tfr_cfg.get("fmin", args.tfr_fmin)))
    set_if_default(args, defaults, "tfr_fmax", float(tfr_cfg.get("fmax", args.tfr_fmax)))
    set_if_default(args, defaults, "tfr_fstep", float(tfr_cfg.get("fstep", args.tfr_fstep)))
    set_if_default(args, defaults, "tfr_method", tfr_cfg.get("method", args.tfr_method))
    set_if_default(args, defaults, "tfr_n_cycles_div", float(tfr_cfg.get("n_cycles_div", args.tfr_n_cycles_div)))
    set_if_default(args, defaults, "tfr_decim", int(tfr_cfg.get("decim", args.tfr_decim)))
    set_if_default(args, defaults, "tfr_time_decim", int(tfr_cfg.get("time_decim", args.tfr_time_decim)))
    b = tfr_cfg.get("baseline", [args.tfr_baseline[0], args.tfr_baseline[1]])
    if isinstance(b, (list, tuple)) and len(b) >= 2:
        set_if_default(args, defaults, "tfr_baseline", [float(b[0]), float(b[1])])
    set_if_default(
        args,
        defaults,
        "tfr_baseline_mode",
        tfr_cfg.get("baseline_mode", tfr_cfg.get("mode", args.tfr_baseline_mode)),
    )

    # Compute
    compute_cfg = cfg.get("compute", {})
    set_if_default(args, defaults, "use_gpu", bool(compute_cfg.get("use_gpu", args.use_gpu)))
    set_if_default(args, defaults, "gpu_device", compute_cfg.get("gpu_device", args.gpu_device))

    # Token map
    if args.token_map is None:
        tm = cfg.get("labels", {}).get("token_map", None)
        if isinstance(tm, dict):
            args.token_map = [f"{k}={v}" for k, v in tm.items()]
        elif isinstance(tm, list):
            args.token_map = tm
        else:
            args.token_map = None

    _finalize_runtime_paths(args, cfg)
    return cfg


def apply_erp_core_preset(args, defaults):
    """Apply ERP CORE-style defaults (TP9/TP10, 0.1–20 Hz, ICA on, individualized thresholds)."""
    if not getattr(args, "erp_core", False):
        return
    # Store for logging
    args._erp_core_preset_enabled = True
    # Preprocessing: ERP CORE uses TP9/TP10 and 0.1 Hz high-pass.
    set_if_default(args, defaults, "reref", "tp9_tp10")
    set_if_default(args, defaults, "l_freq", 0.1)
    # Apply 20 Hz low-pass to align with ERP CORE measurement filtering.
    set_if_default(args, defaults, "h_freq", 20.0)
    # Artifact thresholds: individualized via percentile-based rule.
    set_if_default(args, defaults, "volt_method", "simple")
    set_if_default(args, defaults, "volt_auto_percentile", 97.5)
    set_if_default(args, defaults, "blink_auto_percentile", 99.0)
    # ERP CORE runs ICA by default.
    set_if_default(args, defaults, "ica", "on")


def run_legacy_to_bids_conversion(args, defaults=None, cfg=None) -> list[PipelineRecording]:
    if defaults is None:
        defaults = {}
    if cfg is None:
        cfg = apply_config(args, defaults)

    _finalize_runtime_paths(args, cfg)
    if _input_mode(args, cfg) != "legacy":
        raise ValueError("Legacy-to-BIDS conversion requires --legacy input mode.")

    raw_dir = getattr(args, "raw_dir", None)
    if raw_dir is None:
        raise ValueError("Legacy-to-BIDS conversion requires --raw_dir or paths.raw_dir.")
    bids_root = getattr(args, "conversion_bids_root", None)
    if bids_root is None:
        raise ValueError("Legacy-to-BIDS conversion requires --conversion_bids_root or conversion.bids_output_root.")

    task_label = _legacy_task_label(args, cfg)
    recordings = discover_pipeline_recordings(
        mode="legacy",
        bids_root=None,
        raw_dir=raw_dir,
        subject_csv_dir=getattr(args, "subject_csv_dir", None),
        subjects=getattr(args, "subjects", None),
        sessions=getattr(args, "sessions", None),
        tasks=getattr(args, "tasks", None),
        runs=getattr(args, "runs", None),
        task_label=task_label,
    )
    if not recordings:
        raise RuntimeError(f"No legacy EEG recordings found in {raw_dir}")

    converted = convert_legacy_recordings_to_bids(
        recordings,
        bids_root=bids_root,
        task_label=task_label,
        keep_codes=getattr(args, "behavioral_keep_codes", None),
        standard_codes=getattr(args, "standard_codes", None),
        deviant_codes=getattr(args, "deviant_codes", None),
        drop_eeg_markers_by_gap_s=getattr(args, "drop_eeg_markers_by_gap_s", None),
        auto_drop_to_count=bool(getattr(args, "auto_drop_to_count", 1)),
        overwrite=bool(getattr(args, "conversion_overwrite", 1)),
    )
    args.bids_root = Path(bids_root)
    print(f"Converted legacy dataset -> {args.bids_root}")
    return converted


def run_full_pipeline(args, defaults=None, cfg=None):
    """Run the full EEG processing pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    defaults : dict or None
        Mapping from argument names to their argparse defaults.  Used to
        determine whether CLI arguments were explicitly provided.  If None,
        an empty dict is used and all config values will be applied.
    """
    if defaults is None:
        defaults = {}
    if cfg is None:
        cfg = apply_config(args, defaults)
    _finalize_runtime_paths(args, cfg)
    csv_fallback_dir = (
        None
        if getattr(args, "behavior_csv_fallback_dir", None) in (None, "")
        else Path(args.behavior_csv_fallback_dir)
    )

    ep = EpochParams(
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=(float(args.baseline[0]), float(args.baseline[1])),
    )

    token_map = parse_token_map(args.token_map)

    rows: list[dict] = []
    evokeds_by_group: dict[tuple[str | None, str | None], dict[str, list[mne.Evoked]]] = {}

    # Metrics outputs collected across subjects
    erp_metrics_all: list[pd.DataFrame] = []
    tfr_metrics_all: list[pd.DataFrame] = []
    erp_timeseries_all: list[pd.DataFrame] = []

    input_mode = _input_mode(args, cfg)
    task_label = _legacy_task_label(args, cfg)
    source_dataset = getattr(args, "bids_root", None) if input_mode == "bids" else getattr(args, "raw_dir", None)
    if input_mode == "legacy" and bool(getattr(args, "convert_to_bids", False)):
        recordings = run_legacy_to_bids_conversion(args, defaults=defaults, cfg=cfg)
        source_dataset = getattr(args, "bids_root", None)
    else:
        recordings = discover_pipeline_recordings(
            mode=input_mode,
            bids_root=getattr(args, "bids_root", None),
            raw_dir=getattr(args, "raw_dir", None),
            subject_csv_dir=getattr(args, "subject_csv_dir", None),
            subjects=args.subjects,
            sessions=getattr(args, "sessions", None),
            tasks=getattr(args, "tasks", None),
            runs=getattr(args, "runs", None),
            task_label=task_label,
        )
    dataset_root = _prepare_derivatives_root(args, source_dataset=source_dataset)
    _dataset_metrics_dir(dataset_root)
    if not recordings:
        source_root = getattr(args, "bids_root", None) if input_mode == "bids" else getattr(args, "raw_dir", None)
        mode_label = "BIDS" if input_mode == "bids" else "legacy"
        raise RuntimeError(f"No {mode_label} EEG recordings found in {source_root}")

    std_codes = np.asarray(getattr(args, "standard_codes", []) or [], dtype=int)
    dev_codes = np.asarray(getattr(args, "deviant_codes", []) or [], dtype=int)
    stddev_set = np.r_[std_codes, dev_codes]

    condition_map = getattr(args, "condition_map", None)

    metrics_conditions = getattr(args, "metrics_conditions", None)
    if not metrics_conditions:
        if condition_map:
            metrics_conditions = list(condition_map.keys())
        else:
            metrics_conditions = ["Standard", "Deviant"]

    for recording in recordings:
        raw_path = recording.raw_path
        subj = recording.subject_label
        is_bv = raw_path.suffix.lower() == ".vhdr"
        vmrk = raw_path.with_suffix(".vmrk") if is_bv else None
        subject_base = {
            "subject": subj,
            "session": recording.session_label or "",
            "task": recording.task_id or "",
            "run": recording.run_id or "",
            "raw_file": recording.relative_raw_path,
        }
        _, _, _, behavior_hint = _behavior_inputs_for_recording(recording)

        preproc_path = subject_derivative_path(
            dataset_root,
            recording.entities,
            suffix="eeg",
            extension=".fif",
            desc="preproc",
        )
        epochs_path = subject_derivative_path(
            dataset_root,
            recording.entities,
            suffix="epo",
            extension=".fif",
        )
        aligned_events_path = subject_derivative_path(
            dataset_root,
            recording.entities,
            suffix="events",
            extension=".tsv",
            desc="aligned",
        )
        ica_path = subject_derivative_path(
            dataset_root,
            recording.entities,
            suffix="ica",
            extension=".fif",
            desc="components",
        )

        print(f"\n=== {subj} ===")

        # Always define burst QC fields so the CSV schema is consistent
        burst_qc = {
            "trigger_burst_flag": False,
            "trigger_n_short_iti": 0,
            "trigger_min_iti_s": "",
            "trigger_burst_max_in_window": "",
            "trigger_burst_n_windows_ge_thresh": 0,
            "trigger_burst_params": "",
        }

        if is_bv:
            if not vmrk or not vmrk.exists():
                msg = f"Missing .vmrk for {subj}: {vmrk}"
                if args.on_missing_vmrk == "fail":
                    raise FileNotFoundError(msg)
                if args.on_missing_vmrk == "skip":
                    print("[WARN]", msg, "-> skipping")
                    rows.append(
                        {
                            **subject_base,
                            **burst_qc,
                            "status": "SKIP_MISSING_VMRK",
                            "error": msg,
                        }
                    )
                    continue
                print("[WARN]", msg)

            ok, reason = brainvision_links_ok(raw_path)
            if not ok:
                msg = f"BrainVision link mismatch in {raw_path.name}: {reason}"
                if args.on_bv_link_mismatch == "fail":
                    raise FileNotFoundError(msg)
                print("[WARN]", msg, "-> skipping")
                rows.append(
                    {
                        **subject_base,
                        **burst_qc,
                        "status": "SKIP_BV_LINK_MISMATCH",
                        "error": msg,
                    }
                )
                continue

        raw = read_raw_preprocess(
            raw_path=raw_path,
            montage=args.montage,
            eog_chs=args.eog_chs,
            aux_chs=args.aux_chs,
            reref=args.reref,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            notch=args.notch,
        )

        ica_diag = compute_ica_diagnostics(
            raw,
            blink_proxy_chs=args.blink_proxy_chs,
            blink_threshold_uv=args.blink_threshold_uv,
            blink_win_ms=args.blink_win_ms,
            blink_step_ms=args.blink_step_ms,
        )

        # ---- ICA: optional fit + apply (before event extraction / epoching) ----
        ica_ran = False
        ica_applied = False
        ica_exclude: list[int] = []
        ica_fit_diag: dict = {}
        ica_find_diag: dict = {}
        do_ica = False
        if args.ica == "on":
            do_ica = True
        elif args.ica == "auto":
            rate = float(ica_diag.get("blink_rate_per_min", np.nan))
            proxy_rate = float(ica_diag.get("blink_proxy_rate_per_min", np.nan))
            blink_rate = rate if np.isfinite(rate) and rate > 0 else proxy_rate
            max_corr = float(ica_diag.get("eog_corr_max", np.nan))

            if np.isfinite(blink_rate) and blink_rate >= args.ica_auto_blink_rate_per_min:
                do_ica = True
            elif np.isfinite(max_corr) and max_corr >= args.ica_corr_thresh:
                do_ica = True

        if do_ica:
            ica_params = ICAParams(
                method=args.ica_method,
                n_components=_parse_n_components(args.ica_n_components),
                random_state=args.ica_random_state,
                max_iter=args.ica_max_iter,
                fit_l_freq=args.ica_fit_l_freq,
                fit_h_freq=args.ica_fit_h_freq,
                corr_thresh=args.ica_corr_thresh,
                max_exclude=args.ica_max_exclude,
                decim=args.ica_decim,
            )

            ica_obj, ica_fit_diag = fit_ica(raw, ica_params)
            if ica_obj is None:
                print(f"[WARN] ICA fit failed for {subj}; continuing without ICA.")
            else:
                ica_ran = True
                ica_exclude, ica_find_diag = find_ica_excludes(
                    ica_obj,
                    raw,
                    eog_chs=args.eog_chs,
                    proxy_chs=args.blink_proxy_chs,
                    corr_thresh=args.ica_corr_thresh,
                    max_exclude=args.ica_max_exclude,
                )
                if len(ica_exclude) > 0:
                    raw = apply_ica(raw, ica_obj, ica_exclude)
                    ica_applied = True
                if bool(args.save_ica):
                    ica_obj.save(ica_path, overwrite=True)
                    _write_output_sidecar(
                        ica_path,
                        args,
                        recording,
                        behavior_source=behavior_hint,
                        extra={
                            "Description": "ICA solution fit by eeg-pipeline before epoching.",
                            "ICAExclude": list(ica_exclude),
                            "ICAFitDiagnostics": ica_fit_diag,
                            "ICAFindDiagnostics": ica_find_diag,
                        },
                    )

        # Events from annotations
        events_ann = events_from_annotations_positions(raw)
        markers_pos = events_ann[:, 0].copy()

        # Trigger burst QC (flag only; do not modify markers_pos)
        burst_diag = detect_trigger_bursts(
            markers_pos=markers_pos,
            sfreq=float(raw.info["sfreq"]),
            min_iti_s=0.02,
            burst_win_s=0.25,
            burst_count=5,
        )
        burst_qc = {
            "trigger_burst_flag": bool(burst_diag.get("burst_flag", False)),
            "trigger_n_short_iti": int(burst_diag.get("n_short_iti", 0) or 0),
            "trigger_min_iti_s": burst_diag.get("min_iti_s", ""),
            "trigger_burst_max_in_window": int(burst_diag.get("burst_max_in_window", 1) or 1),
            "trigger_burst_n_windows_ge_thresh": int(burst_diag.get("burst_n_windows_ge_thresh", 0) or 0),
            "trigger_burst_params": burst_diag.get("burst_params", ""),
        }
        if burst_qc["trigger_burst_flag"]:
            print(
                f"[WARN] Trigger burst detected for {subj}: "
                f"short_iti={burst_qc['trigger_n_short_iti']}, "
                f"max_in_window={burst_qc['trigger_burst_max_in_window']}"
            )

        events_tsv, events_json, csv_path, behavior_hint = _behavior_inputs_for_recording(recording)
        try:
            behavior = load_behavioral_events(
                events_tsv=events_tsv,
                events_json=events_json,
                subject_id=recording.subject_id,
                keep_codes=args.behavioral_keep_codes,
                token_map=token_map,
                condition_map=condition_map,
                csv_path=csv_path,
                csv_fallback_dir=csv_fallback_dir,
            )
        except FileNotFoundError as e:
            msg = str(e)
            print("[WARN]", msg, "-> skipping")
            rows.append(
                {
                    **subject_base,
                    **burst_qc,
                    "behavior_source": "missing",
                    "behavior_source_path": str(behavior_hint),
                    "status": "SKIP_MISSING_EVENTS",
                    "error": msg,
                }
            )
            continue

        behavior_source = behavior.source_path
        codes_all = behavior.codes_all
        codes = behavior.codes
        if behavior.samples is not None:
            markers_aligned = np.asarray(behavior.samples, dtype=int)
            diag = {
                "markers_original": int(len(markers_pos)),
                "markers_dropped_by_gap": 0,
                "markers_dropped_by_auto": 0,
            }
        else:
            try:
                markers_aligned, diag = align_marker_positions_to_codes(
                    markers_pos=markers_pos,
                    sfreq=float(raw.info["sfreq"]),
                    codes=codes,
                    gap_s=args.drop_eeg_markers_by_gap_s,
                    auto_drop_to_count=bool(args.auto_drop_to_count),
                )
            except Exception as e:
                msg = f"Alignment failed for {subj}: {e}"
                print("[WARN]", msg, "-> skipping")
                rows.append(
                    {
                        **subject_base,
                        **burst_qc,
                        "behavior_source": behavior.source,
                        "behavior_source_path": str(behavior_source),
                        "status": "SKIP_ALIGNMENT_FAILED",
                        "error": msg,
                    }
                )
                continue

        review_flag = False
        review_reasons = []

        # 1) StimTrak burst
        if burst_qc["trigger_burst_flag"]:
            review_flag = True
            review_reasons.append("trigger_burst")

        # 2) Huge marker excess before alignment (StimTrak spam)
        if diag.get("markers_original", 0) > 2 * len(codes):
            review_flag = True
            review_reasons.append("markers>>behavior")

        # 3) Big auto-drop suggests trigger noise
        if diag.get("markers_dropped_by_auto", 0) >= 50:
            review_flag = True
            review_reasons.append("large_auto_drop")

        # 4) Too few markers vs expected (typically a recording/annotation problem)
        if diag.get("markers_original", 0) < 0.9 * len(codes):
            review_flag = True
            review_reasons.append("markers<behavior")

        events = build_events_from_positions_and_codes(markers_aligned, codes)
        if condition_map is None and (len(std_codes) == 0 or len(dev_codes) == 0):
            raise ValueError("Standard and deviant codes are required when no condition_map is provided.")

        if condition_map is None:
            events_stddev, event_id = select_and_recode_stddev(events, args.standard_codes, args.deviant_codes)
        else:
            events_stddev = events
            event_id = {}

        if condition_map is None and len(events_stddev) == 0:
            msg = f"No standard/deviant events after filtering for {subj}"
            print("[WARN]", msg, "-> skipping")
            rows.append(
                {
                    **subject_base,
                    "behavior_source": behavior.source,
                    "behavior_source_path": str(behavior_source),
                    **burst_qc,
                    "status": "SKIP_NO_STDDEV_EVENTS",
                    "error": msg,
                }
            )
            continue

        events_export = _finalized_events_table(
            behavior.metadata,
            sfreq=float(raw.info["sfreq"]),
            samples=markers_aligned,
            codes=codes,
        )
        trial_levels = None
        if "trial_type" in events_export.columns:
            trial_levels = {
                str(value): str(value)
                for value in sorted(events_export["trial_type"].dropna().astype(str).unique())
            }
        _save_dataframe_with_sidecar(
            events_export,
            aligned_events_path,
            args,
            recording,
            behavior_source=behavior_source,
            description="Aligned event table written by eeg-pipeline in BIDS events format.",
            column_descriptions=_events_json_sidecar(events_export, trial_type_levels=trial_levels),
        )

        if condition_map:
            events_epo, event_id, cond_codes = select_and_filter_conditions(events, condition_map)
            keep_mask = np.isin(events[:, 2], np.asarray(cond_codes, dtype=int))
        else:
            events_epo, event_id = select_and_recode_stddev(events, args.standard_codes, args.deviant_codes)
            keep_mask = np.isin(events[:, 2], stddev_set)

        md_full = behavior.metadata.reset_index(drop=True)

        if len(events_epo) == 0:
            reason = "condition_map" if condition_map else "standard/deviant codes"
            msg = f"No matching events after applying {reason}; skipping."
            print("[WARN]", msg)
            rows.append(
                {
                    **subject_base,
                    "behavior_source": behavior.source,
                    "behavior_source_path": str(behavior_source),
                    **diag,
                    **burst_qc,
                    "n_events_used": int(len(events)),
                    "n_events_kept_stddev": 0,
                    "status": "SKIP_NO_CONDITION_EVENTS",
                    "error": msg,
                }
            )
            raw.save(preproc_path, overwrite=True)
            _write_output_sidecar(
                preproc_path,
                args,
                recording,
                behavior_source=behavior_source,
                extra={"Description": "Preprocessed continuous EEG after filtering and rereferencing."},
            )
            continue

        epochs = make_epochs(raw, events_epo, event_id, ep)
        md = md_full.loc[keep_mask].reset_index(drop=True)
        # Align metadata with epochs that survive MNE's internal dropping
        if len(md) != len(epochs):
            md = md.iloc[epochs.selection].reset_index(drop=True)
        epochs.metadata = md

        epochs_test = epochs.copy().crop(tmin=args.art_test_tmin, tmax=args.art_test_tmax)

        eog_picks = mne.pick_types(epochs_test.info, eog=True, eeg=False)
        blink_threshold_uv = float(args.blink_threshold_uv)
        blink_auto_pct = getattr(args, "blink_auto_percentile", None)
        if blink_auto_pct in ("None", "null"):
            blink_auto_pct = None
        if blink_auto_pct is not None:
            blink_auto_pct = float(blink_auto_pct)
        blink_picks = eog_picks
        if len(blink_picks) == 0:
            proxy = [ch for ch in args.blink_proxy_chs if ch in epochs_test.ch_names]
            if proxy:
                blink_picks = mne.pick_channels(epochs_test.ch_names, include=proxy)
        if blink_auto_pct is not None and len(blink_picks) > 0:
            blink_data = epochs_test.get_data(picks=blink_picks)
            if blink_data.size:
                ptp_max = moving_window_ptp_max(
                    blink_data,
                    sfreq=float(epochs_test.info["sfreq"]),
                    win_ms=args.blink_win_ms,
                    step_ms=args.blink_step_ms,
                )
                if np.isfinite(ptp_max).any():
                    blink_threshold_uv = float(np.nanpercentile(ptp_max, blink_auto_pct))
        blink_bad = np.zeros(len(epochs_test), dtype=bool)

        if len(blink_picks) > 0:
            blink_bad = moving_window_ptp_mask(
                epochs_test.get_data(picks=blink_picks),
                sfreq=float(epochs_test.info["sfreq"]),
                win_ms=args.blink_win_ms,
                step_ms=args.blink_step_ms,
                threshold_uv=blink_threshold_uv,
            )

        eeg_picks = mne.pick_types(epochs_test.info, eeg=True, eog=False)
        volt_method = str(getattr(args, "volt_method", "simple")).lower()
        volt_pos_uv = float(args.volt_pos_uv)
        volt_neg_uv = float(args.volt_neg_uv)
        volt_threshold_uv = float(getattr(args, "volt_threshold_uv", 150.0))
        volt_auto_pct = getattr(args, "volt_auto_percentile", None)
        if volt_auto_pct in ("None", "null"):
            volt_auto_pct = None
        if volt_auto_pct is not None:
            volt_auto_pct = float(volt_auto_pct)
        if volt_auto_pct is not None and len(eeg_picks) > 0:
            eeg_data = epochs_test.get_data(picks=eeg_picks)
            if eeg_data.size:
                max_abs = np.nanmax(np.abs(eeg_data) * 1e6, axis=(1, 2))
                if np.isfinite(max_abs).any():
                    thr_abs = float(np.nanpercentile(max_abs, volt_auto_pct))
                    if volt_method in {"simple", "combined"}:
                        volt_pos_uv = thr_abs
                        volt_neg_uv = -thr_abs
                if volt_method in {"window_ptp", "combined"}:
                    ptp_max = moving_window_ptp_max(
                        eeg_data,
                        sfreq=float(epochs_test.info["sfreq"]),
                        win_ms=float(getattr(args, "volt_win_ms", 200.0)),
                        step_ms=float(getattr(args, "volt_step_ms", 10.0)),
                    )
                    if np.isfinite(ptp_max).any():
                        volt_threshold_uv = float(np.nanpercentile(ptp_max, volt_auto_pct))
        if volt_method == "window_ptp":
            muscle_bad = moving_window_ptp_mask(
                epochs_test.get_data(picks=eeg_picks),
                sfreq=float(epochs_test.info["sfreq"]),
                win_ms=float(getattr(args, "volt_win_ms", 200.0)),
                step_ms=float(getattr(args, "volt_step_ms", 10.0)),
                threshold_uv=volt_threshold_uv,
            )
        elif volt_method == "combined":
            simple_bad = simple_voltage_threshold_mask(
                epochs_test.get_data(picks=eeg_picks),
                pos_limit_uv=volt_pos_uv,
                neg_limit_uv=volt_neg_uv,
            )
            ptp_bad = moving_window_ptp_mask(
                epochs_test.get_data(picks=eeg_picks),
                sfreq=float(epochs_test.info["sfreq"]),
                win_ms=float(getattr(args, "volt_win_ms", 200.0)),
                step_ms=float(getattr(args, "volt_step_ms", 10.0)),
                threshold_uv=volt_threshold_uv,
            )
            muscle_bad = simple_bad | ptp_bad
        else:
            muscle_bad = simple_voltage_threshold_mask(
                epochs_test.get_data(picks=eeg_picks),
                pos_limit_uv=volt_pos_uv,
                neg_limit_uv=volt_neg_uv,
            )

        step_thresh = getattr(args, "volt_step_uv_per_ms", None)
        if step_thresh not in (None, "None", "null"):
            step_bad = step_threshold_mask(
                epochs_test.get_data(picks=eeg_picks),
                sfreq=float(epochs_test.info["sfreq"]),
                threshold_uv_per_ms=float(step_thresh),
            )
            muscle_bad = muscle_bad | step_bad

        threshold_info = {
            "blink_threshold_uv_used": float(blink_threshold_uv),
            "blink_auto_percentile": "" if blink_auto_pct is None else float(blink_auto_pct),
            "volt_pos_uv_used": float(volt_pos_uv),
            "volt_neg_uv_used": float(volt_neg_uv),
            "volt_ptp_threshold_uv_used": (
                float(volt_threshold_uv) if volt_method in {"window_ptp", "combined"} else ""
            ),
            "volt_auto_percentile": "" if volt_auto_pct is None else float(volt_auto_pct),
            "volt_method": volt_method,
        }

        bad = blink_bad | muscle_bad
        bad_idx = np.where(bad)[0].tolist()

        n_before = len(epochs)
        if bad_idx:
            epochs.drop(bad_idx, reason="ARTIFACT_REJECT_MNE")
        n_after = len(epochs)

        if n_after == 0:
            msg = "All epochs dropped after artifact rejection; skipping evoked computation."
            print("[WARN]", msg)
            rows.append(
                {
                    **subject_base,
                    "behavior_source": behavior.source,
                    "behavior_source_path": str(behavior_source),
                    **diag,
                    **burst_qc,
                    **threshold_info,
                    "n_epochs_before_artifact": int(n_before),
                    "n_epochs_final": 0,
                    "status": "SKIP_EMPTY_EPOCHS",
                    "error": msg,
                }
            )
            raw.save(preproc_path, overwrite=True)
            _write_output_sidecar(
                preproc_path,
                args,
                recording,
                behavior_source=behavior_source,
                extra={"Description": "Preprocessed continuous EEG after filtering and rereferencing."},
            )
            continue

        if condition_map:
            n_std = int(np.isin(epochs.events[:, 2], std_codes).sum())
            n_dev = int(np.isin(epochs.events[:, 2], dev_codes).sum())
        else:
            n_std = len(epochs["Standard"])
            n_dev = len(epochs["Deviant"])

        if (not condition_map) and (n_std == 0 or n_dev == 0):
            msg = f"Empty condition after rejection (Standard={n_std}, Deviant={n_dev}); skipping evokeds."
            print("[WARN]", msg)
            rows.append(
                {
                    **subject_base,
                    "behavior_source": behavior.source,
                    "behavior_source_path": str(behavior_source),
                    **diag,
                    **burst_qc,
                    **threshold_info,
                    "n_epochs_before_artifact": int(n_before),
                    "n_epochs_final": int(n_after),
                    "n_standard_final": int(n_std),
                    "n_deviant_final": int(n_dev),
                    "status": "SKIP_EMPTY_CONDITION",
                    "error": msg,
                }
            )
            raw.save(preproc_path, overwrite=True)
            epochs.save(epochs_path, overwrite=True)
            _write_output_sidecar(
                preproc_path,
                args,
                recording,
                behavior_source=behavior_source,
                extra={"Description": "Preprocessed continuous EEG after filtering and rereferencing."},
            )
            _write_output_sidecar(
                epochs_path,
                args,
                recording,
                behavior_source=behavior_source,
                extra={
                    "Description": "Subject-level epochs after artifact rejection.",
                    "EventID": event_id,
                },
            )
            continue

        epoch_reject_rate = (n_before - n_after) / n_before if n_before > 0 else 0.0
        max_rr = getattr(args, "max_reject_rate", None)
        if max_rr is not None and epoch_reject_rate > float(max_rr):
            msg = (
                f"Epoch reject rate {epoch_reject_rate:.3f} exceeds max_reject_rate={float(max_rr):.3f}; "
                "excluding subject from evoked/metrics."
            )
            print("[WARN]", msg)
            rows.append(
                {
                    **subject_base,
                    "behavior_source": behavior.source,
                    "behavior_source_path": str(behavior_source),
                    **diag,
                    **burst_qc,
                    **threshold_info,
                    "n_epochs_before_artifact": int(n_before),
                    "n_epochs_final": int(n_after),
                    "n_standard_final": int(n_std),
                    "n_deviant_final": int(n_dev),
                    "epoch_reject_rate": float(epoch_reject_rate),
                    "status": "SKIP_REJECT_RATE",
                    "error": msg,
                }
            )
            raw.save(preproc_path, overwrite=True)
            epochs.save(epochs_path, overwrite=True)
            _write_output_sidecar(
                preproc_path,
                args,
                recording,
                behavior_source=behavior_source,
                extra={"Description": "Preprocessed continuous EEG after filtering and rereferencing."},
            )
            _write_output_sidecar(
                epochs_path,
                args,
                recording,
                behavior_source=behavior_source,
                extra={
                    "Description": "Subject-level epochs after artifact rejection.",
                    "EventID": event_id,
                },
            )
            continue

        # ------------------------------------------------------------------
        # Metrics (ERP + TFR)
        # ------------------------------------------------------------------
        if int(getattr(args, "metrics", 0)):
            do_erp = bool(getattr(args, "metrics_erp_enabled", True))
            do_tfr = bool(getattr(args, "metrics_tfr_enabled", True))

            if do_erp or do_tfr:
                subject_metrics_dir = _subject_metrics_dir(dataset_root, recording)
                subject_metrics_dir.mkdir(parents=True, exist_ok=True)

            channels = getattr(args, "metrics_channels", None) or ["Fp1", "Fz", "Cz"]
            conds = metrics_conditions

            if do_erp:
                # ERP windows
                erp_windows = _build_erp_windows(args)

                try:
                    diff_label = getattr(args, "difference_label", None)
                    df_erp = compute_erp_metrics(
                        epochs,
                        subject=subj,
                        channels=channels,
                        conditions=conds,
                        windows=erp_windows,
                        compute_mmn=bool(getattr(args, "compute_mmn", 1)),
                        mmn_name=diff_label if diff_label else "DEV_MINUS_STD",
                    )
                    df_erp["subject"] = subj
                    df_erp["task"] = recording.task_id or ""
                    df_erp["session"] = recording.session_label or ""
                    df_erp["run"] = recording.run_id or ""
                    _save_dataframe_with_sidecar(
                        df_erp,
                        subject_derivative_path(
                            dataset_root,
                            recording.entities,
                            suffix="metrics",
                            extension=".tsv",
                            desc="erp",
                        ),
                        args,
                        recording,
                        behavior_source=behavior_source,
                        description="Subject-level ERP metrics computed from derivative epochs.",
                    )
                    erp_metrics_all.append(df_erp)
                except Exception as e:
                    print(f"[WARN] ERP metrics failed for {subj}: {e}")

                if bool(getattr(args, "metrics_erp_timeseries", False)):
                    try:
                        if args.metrics_channels is None:
                            eeg_picks = mne.pick_types(epochs.info, eeg=True, eog=False)
                            ts_channels = [epochs.ch_names[i] for i in eeg_picks]
                        else:
                            ts_channels = channels

                        ts_params = ERPTimeSeriesParams(
                            tmin=float(args.tmin),
                            tmax=float(args.tmax),
                            baseline=(float(args.baseline[0]), float(args.baseline[1])),
                            decim=1,
                        )
                        df_ts = compute_erp_timeseries(
                            epochs,
                            subject=subj,
                            channels=ts_channels,
                            params=ts_params,
                            conditions=conds,
                            include_difference_wave=False,
                        )
                        df_ts["subject"] = subj
                        df_ts["task"] = recording.task_id or ""
                        df_ts["session"] = recording.session_label or ""
                        df_ts["run"] = recording.run_id or ""
                        _save_dataframe_with_sidecar(
                            df_ts,
                            subject_derivative_path(
                                dataset_root,
                                recording.entities,
                                suffix="timeseries",
                                extension=".parquet",
                                desc="erp",
                            ),
                            args,
                            recording,
                            behavior_source=behavior_source,
                            description="Subject-level ERP time series metrics.",
                        )
                        erp_timeseries_all.append(df_ts)
                    except Exception as e:
                        print(f"[WARN] ERP timeseries failed for {subj}: {e}")

            if do_tfr:
                try:
                    tfr_params = TFRParams(
                        fmin=float(getattr(args, "tfr_fmin", 1.0)),
                        fmax=float(getattr(args, "tfr_fmax", 30.0)),
                        fstep=float(getattr(args, "tfr_fstep", 1.0)),
                        method=str(getattr(args, "tfr_method", "multitaper")),
                        n_cycles_div=float(getattr(args, "tfr_n_cycles_div", 10.0)),
                        decim=int(getattr(args, "tfr_decim", 1)),
                        baseline=(
                            float(getattr(args, "tfr_baseline", [-0.1, 0.0])[0]),
                            float(getattr(args, "tfr_baseline", [-0.1, 0.0])[1]),
                        ),
                        mode=str(getattr(args, "tfr_baseline_mode", "logratio")),
                    )
                    df_tfr = compute_tfr_metrics(
                        epochs,
                        subject=subj,
                        channels=channels,
                        conditions=conds,
                        params=tfr_params,
                        tmin=float(getattr(args, "tfr_tmin", -0.2)),
                        tmax=float(getattr(args, "tfr_tmax", 0.6)),
                        time_decim=int(getattr(args, "tfr_time_decim", 1)),
                    )
                    df_tfr["subject"] = subj
                    df_tfr["task"] = recording.task_id or ""
                    df_tfr["session"] = recording.session_label or ""
                    df_tfr["run"] = recording.run_id or ""
                    _save_dataframe_with_sidecar(
                        df_tfr,
                        subject_derivative_path(
                            dataset_root,
                            recording.entities,
                            suffix="metrics",
                            extension=".tsv",
                            desc="tfr",
                        ),
                        args,
                        recording,
                        behavior_source=behavior_source,
                        description="Subject-level TFR metrics computed from derivative epochs.",
                    )
                    tfr_metrics_all.append(df_tfr)
                except Exception as e:
                    print(f"[WARN] TFR metrics failed for {subj}: {e}")

        ica_recommendation = recommend_ica(
            epoch_reject_rate=epoch_reject_rate,
            eog_corr_max=ica_diag.get("eog_corr_max", 0.0),
            blink_rate_per_min=ica_diag.get("blink_rate_per_min", 0.0),
            blink_proxy_rate_per_min=ica_diag.get("blink_proxy_rate_per_min", 0.0),
            epoch_loss_thresh=0.20,
            eog_corr_thresh=args.ica_corr_thresh,
            blink_rate_thresh=args.ica_auto_blink_rate_per_min,
        )

        raw.save(preproc_path, overwrite=True)
        epochs.save(epochs_path, overwrite=True)
        _write_output_sidecar(
            preproc_path,
            args,
            recording,
            behavior_source=behavior_source,
            extra={
                "Description": "Preprocessed continuous EEG after filtering and rereferencing.",
                "ICAApplied": bool(ica_applied),
            },
        )
        _write_output_sidecar(
            epochs_path,
            args,
            recording,
            behavior_source=behavior_source,
            extra={
                "Description": "Subject-level epochs after artifact rejection.",
                "EventID": event_id,
                "EpochRejectRate": float(epoch_reject_rate),
            },
        )

        evoked_conditions = list(event_id.keys())
        evokeds = compute_evokeds(epochs, evoked_conditions)
        for cond, ev in evokeds.items():
            evoked_path = subject_derivative_path(
                dataset_root,
                recording.entities,
                suffix="ave",
                extension=".fif",
                desc=cond.lower(),
            )
            ev.save(evoked_path, overwrite=True)
            _write_output_sidecar(
                evoked_path,
                args,
                recording,
                behavior_source=behavior_source,
                extra={
                    "Description": f"Subject-level evoked average for {cond}.",
                    "Condition": cond,
                    "Nave": getattr(ev, "nave", None),
                },
            )
            group_key = (recording.session_id, recording.task_id)
            evokeds_by_group.setdefault(group_key, {}).setdefault(cond, []).append(ev)

        rows.append(
            {
                **subject_base,
                "sfreq": float(raw.info["sfreq"]),
                "token1": token_map.get("token1"),
                "token2": token_map.get("token2"),
                "behavior_source": behavior.source,
                "behavior_source_path": str(behavior_source),
                "behavioral_codes_total": int(len(codes_all)),
                "behavioral_codes_used": int(len(codes)),
                "behavioral_keep_codes": " ".join(map(str, args.behavioral_keep_codes)) if args.behavioral_keep_codes else "",
                **diag,
                **burst_qc,
                **threshold_info,
                "n_events_used": int(len(events)),
                "n_events_kept_stddev": int(len(events_epo)),
                "n_epochs_before_artifact": int(n_before),
                "n_blink_bad": int(blink_bad.sum()),
                "n_muscle_bad": int(muscle_bad.sum()),
                "n_epochs_dropped": int(n_before - n_after),
                "n_epochs_final": int(n_after),
                "n_standard_final": int(n_std),
                "n_deviant_final": int(n_dev),
                "epoch_reject_rate": float(epoch_reject_rate),
                "eog_corr_max": float(ica_diag.get("eog_corr_max", 0.0) or 0.0),
                "eog_corr_mean": float(ica_diag.get("eog_corr_mean", 0.0) or 0.0),
                "blink_rate_per_min": float(ica_diag.get("blink_rate_per_min", 0.0) or 0.0),
                "blink_proxy_rate_per_min": float(ica_diag.get("blink_proxy_rate_per_min", 0.0) or 0.0),
                "blink_source": ica_diag.get("blink_source", ""),
                "ica_recommended": bool(ica_recommendation.get("ica_recommended", False)),
                "ica_recommend_reason": ica_recommendation.get("ica_recommend_reason", ""),
                "ica_mode": args.ica,
                "ica_ran": bool(ica_ran),
                "ica_applied": bool(ica_applied),
                "ica_exclude": " ".join(map(str, ica_exclude)) if ica_exclude else "",
                **{f"ica_fit_{k}": v for k, v in ica_fit_diag.items()},
                **{f"ica_find_{k}": v for k, v in ica_find_diag.items()},
                "review_flag": review_flag,
                "review_reasons": "+".join(review_reasons),
                "status": "OK",
                "error": "",
            }
        )

        print(
            f"Alignment: markers {diag['markers_original']} -> {len(markers_aligned)} "
            f"(gap_drop={diag['markers_dropped_by_gap']}, auto_drop={diag['markers_dropped_by_auto']})"
        )
        print(
            f"Dropped {n_before - n_after}/{n_before} epochs "
            f"(blink={int(blink_bad.sum())}, muscle={int(muscle_bad.sum())})"
        )
        print(
            f"ICA recommended: {ica_recommendation.get('ica_recommended', False)} "
            f"({ica_recommendation.get('ica_recommend_reason', '')})"
        )

    qc_path = dataset_derivative_path(
        dataset_root,
        suffix="qc",
        extension=".tsv",
        desc="summary",
    )
    write_qc_summary(rows, qc_path)
    write_json(
        derivative_sidecar_path(qc_path),
        {
            "Description": "Dataset-level QC summary for eeg-pipeline derivatives.",
            "GeneratedBy": [{"Name": PIPELINE_NAME, "Version": __version__}],
        },
    )

    if erp_metrics_all:
        _save_dataframe_with_sidecar(
            pd.concat(erp_metrics_all, ignore_index=True),
            dataset_derivative_path(dataset_root, suffix="metrics", extension=".tsv", desc="erp"),
            args,
            None,
            behavior_source=None,
            description="Dataset-level ERP metrics aggregated across processed subjects.",
        )
    if erp_timeseries_all:
        _save_dataframe_with_sidecar(
            pd.concat(erp_timeseries_all, ignore_index=True),
            dataset_derivative_path(dataset_root, suffix="timeseries", extension=".parquet", desc="erp"),
            args,
            None,
            behavior_source=None,
            description="Dataset-level ERP time series aggregated across processed subjects.",
        )
    if tfr_metrics_all:
        _save_dataframe_with_sidecar(
            pd.concat(tfr_metrics_all, ignore_index=True),
            dataset_derivative_path(dataset_root, suffix="metrics", extension=".tsv", desc="tfr"),
            args,
            None,
            behavior_source=None,
            description="Dataset-level TFR metrics aggregated across processed subjects.",
        )

    if not any(evokeds_by_group.values()):
        print("\n[WARN] No successful subjects to grand-average. Writing QC summary only.")
        print(f"Saved QC summary -> {qc_path}")
        return

    for (ses, task), evoked_map in evokeds_by_group.items():
        ga_by_cond = grand_averages(evoked_map)
        group_entities = {}
        if ses:
            group_entities["ses"] = ses
        if task:
            group_entities["task"] = task
        for cond, ga in ga_by_cond.items():
            ga_path = dataset_derivative_path(
                dataset_root,
                entities=group_entities,
                suffix="ave",
                extension=".fif",
                desc=f"grandaverage-{cond.lower()}",
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

    print(f"\nSaved QC summary -> {qc_path}")
    print(f"Saved derivatives -> {dataset_root}")


def _subject_from_epochs_path(p: Path) -> str:
    return source_basename_from_derivative_path(p)


def run_metrics_only(args):
    """Compute ERP/TFR metrics from existing derivative epochs."""
    _finalize_runtime_paths(args)
    dataset_root = _prepare_derivatives_root(args)
    files = sorted(dataset_root.rglob("*_epo.fif"))
    if not files:
        raise RuntimeError(f"No epochs found in {dataset_root} (expected *_epo.fif).")

    do_erp = bool(getattr(args, "metrics_erp_enabled", True))
    do_tfr = bool(getattr(args, "metrics_tfr_enabled", True))

    if not (do_erp or do_tfr):
        print("[WARN] Metrics requested but both ERP and TFR are disabled in config.")
        return

    channels = getattr(args, "metrics_channels", None) or ["Fp1", "Fz", "Cz"]
    metrics_conditions = getattr(args, "metrics_conditions", None)
    if not metrics_conditions:
        cond_map = getattr(args, "condition_map", None)
        if cond_map:
            metrics_conditions = list(cond_map.keys())
        else:
            metrics_conditions = ["Standard", "Deviant"]

    erp_windows = None
    if do_erp:
        erp_windows = _build_erp_windows(args)

    tfr_params = None
    if do_tfr:
        tfr_params = TFRParams(
            fmin=float(getattr(args, "tfr_fmin", 1.0)),
            fmax=float(getattr(args, "tfr_fmax", 30.0)),
            fstep=float(getattr(args, "tfr_fstep", 1.0)),
            method=str(getattr(args, "tfr_method", "multitaper")),
            n_cycles_div=float(getattr(args, "tfr_n_cycles_div", 10.0)),
            decim=int(getattr(args, "tfr_decim", 1)),
            baseline=(
                float(getattr(args, "tfr_baseline", [-0.1, 0.0])[0]),
                float(getattr(args, "tfr_baseline", [-0.1, 0.0])[1]),
            ),
            mode=str(getattr(args, "tfr_baseline_mode", "logratio")),
        )

    erp_metrics_all: list[pd.DataFrame] = []
    tfr_metrics_all: list[pd.DataFrame] = []
    erp_timeseries_all: list[pd.DataFrame] = []

    for p in files:
        source_basename = _subject_from_epochs_path(p)
        entities = parse_bids_entities_like_name(source_basename)
        subj = f"sub-{entities['sub']}"
        loaded = load_epochs(p)
        epochs = loaded.epochs

        if do_erp and erp_windows is not None:
            try:
                diff_label = getattr(args, "difference_label", None)
                df_erp = compute_erp_metrics(
                    epochs,
                    subject=subj,
                    channels=channels,
                    conditions=metrics_conditions,
                    windows=erp_windows,
                    compute_mmn=bool(getattr(args, "compute_mmn", 1)),
                    mmn_name=diff_label if diff_label else "DEV_MINUS_STD",
                )
                df_erp["subject"] = subj
                _save_dataframe_with_sidecar(
                    df_erp,
                    subject_derivative_path(
                        dataset_root,
                        entities,
                        suffix="metrics",
                        extension=".tsv",
                        desc="erp",
                    ),
                    args,
                    None,
                    behavior_source=None,
                    description="Subject-level ERP metrics computed from derivative epochs.",
                )
                erp_metrics_all.append(df_erp)
            except Exception as e:
                print(f"[WARN] ERP metrics failed for {subj}: {e}")

        if bool(getattr(args, "metrics_erp_timeseries", False)):
            try:
                if args.metrics_channels is None:
                    eeg_picks = mne.pick_types(epochs.info, eeg=True, eog=False)
                    ts_channels = [epochs.ch_names[i] for i in eeg_picks]
                else:
                    ts_channels = channels

                ts_params = ERPTimeSeriesParams(
                    tmin=float(args.tmin),
                    tmax=float(args.tmax),
                    baseline=(float(args.baseline[0]), float(args.baseline[1])),
                    decim=1,
                )
                df_ts = compute_erp_timeseries(
                    epochs,
                    subject=subj,
                    channels=ts_channels,
                    params=ts_params,
                    conditions=metrics_conditions,
                    include_difference_wave=False,
                )
                df_ts["subject"] = subj
                _save_dataframe_with_sidecar(
                    df_ts,
                    subject_derivative_path(
                        dataset_root,
                        entities,
                        suffix="timeseries",
                        extension=".parquet",
                        desc="erp",
                    ),
                    args,
                    None,
                    behavior_source=None,
                    description="Subject-level ERP time series metrics.",
                )
                erp_timeseries_all.append(df_ts)
            except Exception as e:
                print(f"[WARN] ERP timeseries failed for {subj}: {e}")

        if do_tfr and tfr_params is not None:
            try:
                df_tfr = compute_tfr_metrics(
                    epochs,
                    subject=subj,
                    channels=channels,
                    conditions=metrics_conditions,
                    params=tfr_params,
                    tmin=float(getattr(args, "tfr_tmin", -0.2)),
                    tmax=float(getattr(args, "tfr_tmax", 0.6)),
                    time_decim=int(getattr(args, "tfr_time_decim", 1)),
                )
                df_tfr["subject"] = subj
                _save_dataframe_with_sidecar(
                    df_tfr,
                    subject_derivative_path(
                        dataset_root,
                        entities,
                        suffix="metrics",
                        extension=".tsv",
                        desc="tfr",
                    ),
                    args,
                    None,
                    behavior_source=None,
                    description="Subject-level TFR metrics computed from derivative epochs.",
                )
                tfr_metrics_all.append(df_tfr)
            except Exception as e:
                print(f"[WARN] TFR metrics failed for {subj}: {e}")

    if erp_metrics_all:
        _save_dataframe_with_sidecar(
            pd.concat(erp_metrics_all, ignore_index=True),
            dataset_derivative_path(dataset_root, suffix="metrics", extension=".tsv", desc="erp"),
            args,
            None,
            behavior_source=None,
            description="Dataset-level ERP metrics aggregated across processed subjects.",
        )
    if erp_timeseries_all:
        _save_dataframe_with_sidecar(
            pd.concat(erp_timeseries_all, ignore_index=True),
            dataset_derivative_path(dataset_root, suffix="timeseries", extension=".parquet", desc="erp"),
            args,
            None,
            behavior_source=None,
            description="Dataset-level ERP time series aggregated across processed subjects.",
        )
    if tfr_metrics_all:
        _save_dataframe_with_sidecar(
            pd.concat(tfr_metrics_all, ignore_index=True),
            dataset_derivative_path(dataset_root, suffix="metrics", extension=".tsv", desc="tfr"),
            args,
            None,
            behavior_source=None,
            description="Dataset-level TFR metrics aggregated across processed subjects.",
        )


def _resolve_figure_time_window(args) -> tuple[float, float]:
    if args.figure_time_window is not None:
        return float(args.figure_time_window[0]), float(args.figure_time_window[1])
    if getattr(args, "erp_window", None):
        w = args.erp_window[0]
        return float(w[1]), float(w[2])
    return float(args.tmin), float(args.tmax)


def _resolve_figure_freq_band(args) -> tuple[float, float] | None:
    if args.figure_freq_band is not None:
        return float(args.figure_freq_band[0]), float(args.figure_freq_band[1])
    return float(getattr(args, "tfr_fmin", 1.0)), float(getattr(args, "tfr_fmax", 30.0))


def _build_erp_windows(args) -> list[ERPWindow]:
    windows: list[ERPWindow] = []
    if getattr(args, "erp_window", None):
        windows = [
            ERPWindow(name=w[0], tmin=float(w[1]), tmax=float(w[2]))
            for w in args.erp_window
        ]
        return windows

    if bool(getattr(args, "compute_mmn", 0)):
        windows.append(ERP_WINDOWS["MMN"])
    if bool(getattr(args, "compute_p300", 0)):
        windows.append(ERP_WINDOWS["P300"])

    return windows


def _prompt_yes_no(msg: str) -> bool:
    if not sys.stdin.isatty():
        return False
    resp = input(msg).strip().lower()
    return resp in {"y", "yes"}


def run_plot_figures(args):
    from eeg_pipeline.viz import paper_figures

    _finalize_runtime_paths(args)
    dataset_root = _pipeline_dataset_root(args)
    _dataset_metrics_dir(dataset_root)
    fig_dir = Path(args.figures_out_dir) if args.figures_out_dir else dataset_root / "figures"

    erp_parquet = dataset_derivative_path(dataset_root, suffix="timeseries", extension=".parquet", desc="erp")
    tfr_file = dataset_derivative_path(dataset_root, suffix="metrics", extension=".tsv", desc="tfr")

    erp_exists = erp_parquet.exists()
    tfr_exists = tfr_file.exists()

    if not erp_exists and not tfr_exists:
        raise FileNotFoundError(
            f"No metrics found for plotting. Expected {erp_parquet} and/or {tfr_file}."
        )

    time_window = _resolve_figure_time_window(args)
    freq_band = _resolve_figure_freq_band(args) if tfr_exists else None

    argv = [
        "--out_dir", str(fig_dir),
        "--time_window", str(time_window[0]), str(time_window[1]),
    ]
    if erp_exists:
        argv += ["--erp_parquet", str(erp_parquet)]
    if tfr_exists and freq_band is not None:
        argv += [
            "--tfr_file", str(tfr_file),
            "--freq_band", str(freq_band[0]), str(freq_band[1]),
        ]
    if args.figure_diff_heatmap:
        argv.append("--diff_heatmap")
    if args.figure_channels:
        argv += ["--channels", *args.figure_channels]

    paper_figures.main(argv)

def prepare_output_dirs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_derivatives_dataset(out_dir, pipeline_version=__version__)

def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    ap.add_argument(
        "--erp-core",
        dest="erp_core",
        action="store_true",
        help="Use ERP CORE-style defaults (TP9/TP10, 0.1–20 Hz, ICA on, individualized thresholds).",
    )
    ap.add_argument("--process_data", action="store_true", help="Process EEG inputs into BIDS-derivative epochs/evokeds/QC")
    ap.add_argument("--get_metrics", action="store_true", help="Compute ERP/TFR metrics from derivative epochs")
    ap.add_argument("--plot_figures", action="store_true", help="Generate figures from aggregated derivative metrics")
    ap.add_argument(
        "--legacy",
        action="store_true",
        help="Use the original lab layout instead of BIDS input discovery. BIDS is the default.",
    )
    ap.add_argument("--bids_root", help="Root of an input BIDS EEG dataset")
    ap.add_argument("--raw_dir", help="Legacy raw EEG directory used with --legacy")
    ap.add_argument("--subject_csv_dir", help="Optional legacy subject CSV directory used with --legacy")
    ap.add_argument("--derivatives_root", help="Root derivatives folder that will contain derivatives/eeg-pipeline")
    ap.add_argument("--sourcedata_root", default=None, help="Optional sourcedata root associated with the BIDS dataset")
    ap.add_argument(
        "--task_label",
        default=None,
        help="Legacy task label used when raw filenames do not already include task-<label>.",
    )
    ap.add_argument(
        "--behavior_csv_fallback_dir",
        default=None,
        help="Optional fallback directory containing subject CSV files when source events.tsv is unavailable.",
    )
    ap.add_argument(
        "--convert_to_bids",
        action="store_true",
        help="In legacy mode, convert the discovered dataset into BIDS before processing. If no other stage flags are set, conversion runs and exits.",
    )
    ap.add_argument(
        "--conversion_bids_root",
        default=None,
        help="Output root for legacy-to-BIDS conversion.",
    )
    ap.add_argument(
        "--conversion_overwrite",
        type=int,
        default=1,
        help="Overwrite converted BIDS files when --convert_to_bids is enabled (1=yes,0=no).",
    )
    ap.add_argument(
        "--summarize_one_file",
        default=None,
        help="If provided, summarize this raw EEG file (.vhdr or .set) and exit.",
    )

    ap.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration where available (MNE/CuPy).")
    ap.add_argument("--gpu_device", type=int, default=None, help="Optional GPU device index (default: first visible).")

    ap.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject filters (01 or sub-01). If omitted, runs all discovered subjects.",
    )
    ap.add_argument("--sessions", nargs="*", default=None, help="Optional session filters (e.g., 01 or ses-01).")
    ap.add_argument("--tasks", nargs="*", default=None, help="Optional task filters.")
    ap.add_argument("--runs", nargs="*", default=None, help="Optional run filters.")

    ap.add_argument(
        "--on_missing_vmrk",
        choices=["warn", "skip", "fail"],
        default="warn",
        help="What to do if .vmrk is missing next to .vhdr (default: warn).",
    )

    ap.add_argument("--montage", default="standard_1020", help="Montage name")
    ap.add_argument(
        "--reref",
        default="average",
        choices=["average", "none", "p9_p10", "tp9_tp10", "mastoids"],
        help="EEG re-reference mode (average, none, or p9_p10/tp9_tp10 mastoids).",
    )
    ap.add_argument("--l_freq", type=float, default=0.1, help="High-pass Hz")
    ap.add_argument("--h_freq", type=float, default=30.0, help="Low-pass Hz")
    ap.add_argument("--notch", type=float, nargs="*", default=[60.0], help="Notch freqs Hz")

    ap.add_argument("--tmin", type=float, default=-0.2, help="Epoch start (s)")
    ap.add_argument("--tmax", type=float, default=0.6, help="Epoch end (s)")
    ap.add_argument("--baseline", type=float, nargs=2, default=(-0.2, 0.0), help="Baseline (s s)")

    ap.add_argument("--eog_chs", nargs="*", default=[], help="EOG channel names (if present)")
    ap.add_argument("--aux_chs", nargs="*", default=["AUX"], help="Aux channels to drop")

    ap.add_argument(
        "--blink_proxy_chs",
        nargs="*",
        default=["Fp1"],
        help="Frontal EEG channels to use as blink proxy if no EOG channels exist (default: Fp1).",
    )

    ap.add_argument(
        "--behavioral_keep_codes",
        nargs="*",
        type=int,
        default=[110, 111, 210, 211],
        help="Keep only these numeric codes from source events.tsv (or an explicit CSV fallback) when aligning to EEG markers.",
    )
    ap.add_argument(
        "--drop_eeg_markers_by_gap_s",
        type=float,
        default=None,
        help="Optional gap threshold heuristic (seconds) to drop likely boundary markers before auto-drop-to-count.",
    )
    ap.add_argument(
        "--auto_drop_to_count",
        type=int,
        default=1,
        help="If EEG markers > behavioral codes used, auto-drop extra markers to match count (1=yes,0=no).",
    )

    ap.add_argument("--standard_codes", nargs="*", type=int, default=[110, 210], help="Codes considered Standard")
    ap.add_argument("--deviant_codes", nargs="*", type=int, default=[111, 211], help="Codes considered Deviant")

    ap.add_argument(
        "--token_map",
        nargs="*",
        default=None,
        help="Optional token labeling. Either: '--token_map EH IH' or '--token_map Token1=EH Token2=IH' (or mix).",
    )

    # Artifact settings
    ap.add_argument("--art_test_tmin", type=float, default=-0.2)
    ap.add_argument("--art_test_tmax", type=float, default=0.3)
    ap.add_argument("--blink_threshold_uv", type=float, default=75.0)
    ap.add_argument("--blink_win_ms", type=float, default=200.0)
    ap.add_argument("--blink_step_ms", type=float, default=10.0)
    ap.add_argument(
        "--blink_auto_percentile",
        type=float,
        default=None,
        help="Optional per-subject percentile for blink peak-to-peak threshold (e.g., 99).",
    )
    ap.add_argument("--volt_pos_uv", type=float, default=150.0)
    ap.add_argument("--volt_neg_uv", type=float, default=-150.0)
    ap.add_argument(
        "--volt_method",
        default="simple",
        choices=["simple", "window_ptp", "combined"],
        help="EEG artifact rejection method (simple threshold, windowed peak-to-peak, or combined).",
    )
    ap.add_argument("--volt_threshold_uv", type=float, default=150.0)
    ap.add_argument("--volt_win_ms", type=float, default=200.0)
    ap.add_argument("--volt_step_ms", type=float, default=10.0)
    ap.add_argument(
        "--volt_step_uv_per_ms",
        type=float,
        default=None,
        help="Optional voltage step threshold (uV/ms). If set, epochs exceeding this step are rejected.",
    )
    ap.add_argument(
        "--volt_auto_percentile",
        type=float,
        default=None,
        help="Optional per-subject percentile for EEG voltage thresholds (e.g., 97.5).",
    )
    ap.add_argument(
        "--max_reject_rate",
        type=float,
        default=None,
        help="If set, skip evokeds/metrics when epoch reject rate exceeds this fraction (e.g., 0.5).",
    )

    # --- ICA controls ---
    ap.add_argument(
        "--ica",
        choices=["off", "auto", "on"],
        default="off",
        help="ICA mode: off (default), auto (gate by blink rate), or on (always run ICA).",
    )
    ap.add_argument("--ica_method", default="fastica", choices=["fastica", "picard", "infomax"])
    ap.add_argument(
        "--ica_n_components",
        default="0.99",
        type=str,
        help="ICA n_components: float variance fraction (e.g., 0.99) or int (e.g., 20).",
    )
    ap.add_argument("--ica_random_state", default=97, type=int)
    ap.add_argument("--ica_max_iter", default=512, type=int)
    ap.add_argument(
        "--ica_fit_l_freq",
        default=1.0,
        type=float,
        help="High-pass used only for ICA fitting (recommended 1.0).",
    )
    ap.add_argument("--ica_fit_h_freq", default=None, type=float, help="Optional low-pass used only for ICA fitting.")
    ap.add_argument("--ica_decim", default=3, type=int, help="Decimation for ICA fit speed (3 is a good default).")
    ap.add_argument("--ica_corr_thresh", default=0.30, type=float, help="Proxy correlation threshold for excluding components.")
    ap.add_argument("--ica_max_exclude", default=3, type=int, help="Max # components to exclude.")
    ap.add_argument(
        "--ica_auto_blink_rate_per_min",
        default=15.0,
        type=float,
        help="If --ica auto, run ICA when blink rate >= this threshold (per minute).",
    )
    ap.add_argument("--save_ica", default=1, type=int, help="Save ICA object into the BIDS derivatives tree (1=yes,0=no).")

    ap.add_argument(
        "--on_bv_link_mismatch",
        choices=["skip", "fail"],
        default="skip",
        help="What to do if a .vhdr references a missing MarkerFile/DataFile (default: skip).",
    )

    # --- Metrics controls (ERP + TFR) ---
    ap.add_argument(
        "--metrics",
        type=int,
        default=1,
        help="Compute ERP/TFR metrics and write them into the derivatives dataset (1=yes,0=no).",
    )
    ap.add_argument(
        "--metrics_channels",
        nargs="+",
        default=None,
        help="Channels used for metrics (default uses config or a small fronto-central set).",
    )
    ap.add_argument(
        "--metrics_conditions",
        nargs="+",
        default=None,
        help="Condition labels to use for ERP/TFR metrics (must exist in epochs.event_id).",
    )
    ap.add_argument(
        "--erp_window",
        nargs=3,
        action="append",
        default=None,
        metavar=("NAME", "TMIN", "TMAX"),
        help="Add an ERP window, e.g. --erp_window MMN_150_250 0.15 0.25. Can be repeated.",
    )
    ap.add_argument(
        "--compute_mmn",
        type=int,
        default=1,
        help="If 1, include the default MMN window (when none specified) and compute Deviant-Standard difference.",
    )
    ap.add_argument(
        "--difference_label",
        default=None,
        help="Optional label for the Deviant–Standard difference wave (default: DEV_MINUS_STD).",
    )
    ap.add_argument(
        "--compute_p300",
        type=int,
        default=0,
        help="If 1, include the default P300 window when ERP windows are not otherwise specified.",
    )

    # TFR settings (kept simple; can be overridden in config)
    ap.add_argument("--tfr_tmin", type=float, default=-0.2)
    ap.add_argument("--tfr_tmax", type=float, default=0.6)
    ap.add_argument("--tfr_fmin", type=float, default=1.0)
    ap.add_argument("--tfr_fmax", type=float, default=30.0)
    ap.add_argument("--tfr_fstep", type=float, default=1.0)
    ap.add_argument("--tfr_method", default="multitaper", choices=["multitaper", "morlet"])
    ap.add_argument("--tfr_n_cycles_div", type=float, default=10.0)
    ap.add_argument("--tfr_decim", type=int, default=1)
    ap.add_argument(
        "--tfr_time_decim",
        type=int,
        default=1,
        help="Downsample TFR time points in metrics output (1 = no downsample).",
    )
    ap.add_argument("--tfr_baseline", nargs=2, type=float, default=[-0.1, 0.0])
    ap.add_argument("--tfr_baseline_mode", default="logratio")

    # --- Figure controls ---
    ap.add_argument("--figure_time_window", nargs=2, type=float, default=None, metavar=("TMIN", "TMAX"))
    ap.add_argument("--figure_freq_band", nargs=2, type=float, default=None, metavar=("FMIN", "FMAX"))
    ap.add_argument("--figure_diff_heatmap", action="store_true", help="Add deviant-standard heatmap")
    ap.add_argument("--figure_channels", nargs="+", default=None, help="Optional channel subset for ERP plots")
    ap.add_argument("--figures_out_dir", default=None, help="Output directory for figures (default: derivatives/eeg-pipeline/figures)")
    return ap


# -----------------------------------------------------------------------------
# Default handling helpers
#
# To allow command‑line flags to override YAML/JSON configuration values
# cleanly, we record the argparse defaults once up front.  See
# ``run_full_pipeline`` for how these defaults are used together with the
# ``set_if_default`` helper.
def build_defaults(parser: argparse.ArgumentParser) -> dict:
    """Return a mapping from argument name to its argparse default.

    The ``defaults`` dict allows us to detect whether the user set a flag
    explicitly on the command line (in which case ``args.<field>`` will
    differ from the default) or left it unspecified (in which case we can
    safely override it with the value from the config file).
    """
    defaults: dict = {}
    for action in parser._actions:
        if action.dest != "help":
            defaults[action.dest] = action.default
    return defaults


def main(argv=None):
    ap = build_arg_parser()
    # Collect defaults before parsing arguments.  These defaults let us
    # distinguish CLI‑provided arguments from those left unspecified.
    defaults = build_defaults(ap)
    args = ap.parse_args(argv)
    stages_requested = bool(args.process_data or args.get_metrics or args.plot_figures)

    if args.summarize_one_file:
        apply_erp_core_preset(args, defaults)
        cfg = apply_config(args, defaults)
        _finalize_runtime_paths(args, cfg)
        summarize_one_file(args, Path(args.summarize_one_file))
        return

    if not stages_requested:
        if bool(getattr(args, "convert_to_bids", False)):
            args.process_data = False
            args.get_metrics = False
            args.plot_figures = False
        else:
            # Default behavior: process data + metrics
            args.process_data = True
            args.get_metrics = True
            args.plot_figures = False

    # Apply ERP CORE preset before config so it can override config defaults.
    apply_erp_core_preset(args, defaults)

    # Apply config once for all stages
    cfg = apply_config(args, defaults)
    _finalize_runtime_paths(args, cfg)
    if not stages_requested and bool(getattr(args, "convert_to_bids", False)):
        args.process_data = False
        args.get_metrics = False
        args.plot_figures = False

    if getattr(args, "_erp_core_preset_enabled", False):
        print("[ERP-CORE] preset enabled")
        print(
            "[ERP-CORE] reref={reref} l_freq={l_freq} h_freq={h_freq} "
            "volt_method={volt_method} volt_auto_percentile={volt_auto} "
            "blink_auto_percentile={blink_auto} ica={ica}".format(
                reref=getattr(args, "reref", ""),
                l_freq=getattr(args, "l_freq", ""),
                h_freq=getattr(args, "h_freq", ""),
                volt_method=getattr(args, "volt_method", ""),
                volt_auto=getattr(args, "volt_auto_percentile", ""),
                blink_auto=getattr(args, "blink_auto_percentile", ""),
                ica=getattr(args, "ica", ""),
            )
        )

    gpu_status = configure_gpu(bool(args.use_gpu), device=args.gpu_device)
    if args.use_gpu:
        cap_msg = format_capability_report(capability_report())
        if cap_msg:
            print(cap_msg)
        if gpu_status["enabled"]:
            print(
                "[GPU] enabled (backend="
                f"{gpu_status['backend']}, mne_cuda={gpu_status['mne_cuda']}, "
                f"cupy={gpu_status['cupy']})"
            )
        else:
            print(
                "[WARN] GPU requested but not available; falling back to CPU "
                f"(mne_cuda={gpu_status['mne_cuda']}, cupy={gpu_status['cupy']})"
            )

    if args.plot_figures:
        # Ensure ERP time-series is available for plotting
        args.metrics_erp_timeseries = True

    if bool(getattr(args, "convert_to_bids", False)) and not (
        args.process_data or args.get_metrics or args.plot_figures
    ):
        run_legacy_to_bids_conversion(args, defaults=defaults, cfg=cfg)
        return

    if args.process_data:
        if not args.get_metrics:
            args.metrics = 0
        else:
            args.metrics = 1
        run_full_pipeline(args, defaults=defaults, cfg=cfg)
    elif args.get_metrics:
        run_metrics_only(args)

    if args.plot_figures:
        dataset_root = _pipeline_dataset_root(args)
        erp_parquet = dataset_derivative_path(dataset_root, suffix="timeseries", extension=".parquet", desc="erp")
        tfr_file = dataset_derivative_path(dataset_root, suffix="metrics", extension=".tsv", desc="tfr")

        missing = []
        if not erp_parquet.exists():
            missing.append(str(erp_parquet))
        if not tfr_file.exists():
            missing.append(str(tfr_file))

        if missing:
            print(f"[WARN] Missing figure inputs: {', '.join(missing)}")
            if _prompt_yes_no("Run full pipeline now? [y/N] "):
                args.process_data = True
                args.get_metrics = True
                args.metrics = 1
                run_full_pipeline(args, defaults=defaults, cfg=cfg)
            else:
                print("[WARN] Proceeding with available metrics only.")

        run_plot_figures(args)


if __name__ == "__main__":
    main()
