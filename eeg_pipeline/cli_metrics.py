from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd

from .cli_common import (
    ERP_WINDOWS,
    ERPTimeSeriesParams,
    ERPWindow,
    TFRParams,
    _finalize_runtime_paths,
    _prepare_derivatives_root,
    _save_dataframe_with_sidecar,
    compute_erp_metrics,
    compute_erp_timeseries,
    compute_tfr_metrics,
    dataset_derivative_path,
    filter_derivative_paths,
    load_epochs,
    parse_bids_entities_like_name,
    source_basename_from_derivative_path,
    subject_derivative_path,
)


def _subject_from_epochs_path(p: Path) -> str:
    return source_basename_from_derivative_path(p)


def run_metrics_only(args):
    """Compute ERP/TFR metrics from existing derivative epochs."""
    _finalize_runtime_paths(args)
    dataset_root = _prepare_derivatives_root(args)
    discovered = sorted(dataset_root.rglob("*_epo.fif"))
    if not discovered:
        raise RuntimeError(f"No epochs found in {dataset_root} (expected *_epo.fif).")

    # Honor the same entity filters as --process_data. Without this the metrics
    # stage silently recomputes every subject, so a per-subject invocation would
    # duplicate work and race other jobs on the dataset-level outputs.
    files = filter_derivative_paths(
        discovered,
        subjects=getattr(args, "subjects", None),
        sessions=getattr(args, "sessions", None),
        tasks=getattr(args, "tasks", None),
        runs=getattr(args, "runs", None),
    )
    if not files:
        raise RuntimeError(
            f"No epochs in {dataset_root} matched the requested "
            f"subjects/sessions/tasks/runs filters ({len(discovered)} found before filtering)."
        )

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


