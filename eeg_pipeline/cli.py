from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mne
import numpy as np

from . import metrics_runner as _metrics_runner
from . import pipeline_config as _pipeline_config
from . import pipeline_runner as _pipeline_runner
from . import summary_runner as _summary_runner
from .align import (
    align_marker_positions_to_codes,
    detect_trigger_bursts,
    keep_by_gap_heuristic,
    marker_gap_stats,
)
from .artifacts import (
    moving_window_ptp_mask,
    moving_window_ptp_max,
    simple_voltage_threshold_mask,
    step_threshold_mask,
)
from .behavior import (
    clean_eventcodes,
    filter_codes,
    read_eventcodes_from_subject_csv,
    subject_number_from_stem,
)
from .epoching import (
    build_events_from_positions_and_codes,
    make_epochs,
    select_and_filter_conditions,
    select_and_recode_stddev,
)
from .evoked import compute_evokeds, grand_averages
from .figure_runner import resolve_figure_freq_band as _resolve_figure_freq_band
from .figure_runner import resolve_figure_time_window as _resolve_figure_time_window
from .figure_runner import run_plot_figures
from .gpu import capability_report, format_capability_report
from .gpu import configure as configure_gpu
from .ica import apply_ica, find_ica_excludes, fit_ica
from .ica_diagnostics import compute_ica_diagnostics, recommend_ica
from .io_brainvision import (
    _bv_get,
    brainvision_links_ok,
    events_from_annotations_positions,
    parse_vmrk_markers,
    read_raw_preprocess,
)
from .metrics import compute_erp_metrics, compute_tfr_metrics, load_epochs
from .metrics.erp_timeseries import compute_erp_timeseries
from .metrics.erp_windows import ERP_WINDOWS
from .metrics_runner import build_erp_windows as _build_erp_windows
from .metrics_runner import subject_from_epochs_path as _subject_from_epochs_path
from .pipeline_config import apply_erp_core_preset, load_config, set_if_default
from .pipeline_runner import _parse_n_components, prepare_output_dirs
from .qc import write_qc_summary
from .schema import derive_metadata_from_condition_map, derive_metadata_v1, parse_token_map

_COMPAT_EXPORTS = (
    subject_number_from_stem,
    _resolve_figure_freq_band,
    _resolve_figure_time_window,
    _bv_get,
    ERP_WINDOWS,
    _subject_from_epochs_path,
    set_if_default,
    _parse_n_components,
    prepare_output_dirs,
)

_MISSING = object()


def apply_config(args, defaults=None):
    original_loader = _pipeline_config.load_config
    _pipeline_config.load_config = load_config
    try:
        return _pipeline_config.apply_config(args, defaults)
    finally:
        _pipeline_config.load_config = original_loader


def run_metrics_only(args):
    if not hasattr(args, "allow_metric_failures"):
        args.allow_metric_failures = True
    original = {
        "build_erp_windows": _metrics_runner.build_erp_windows,
        "load_epochs": _metrics_runner.load_epochs,
        "compute_erp_metrics": _metrics_runner.compute_erp_metrics,
        "compute_erp_timeseries": _metrics_runner.compute_erp_timeseries,
        "compute_tfr_metrics": _metrics_runner.compute_tfr_metrics,
    }
    _metrics_runner.build_erp_windows = _build_erp_windows
    _metrics_runner.load_epochs = load_epochs
    _metrics_runner.compute_erp_metrics = compute_erp_metrics
    _metrics_runner.compute_erp_timeseries = compute_erp_timeseries
    _metrics_runner.compute_tfr_metrics = compute_tfr_metrics
    try:
        return _metrics_runner.run_metrics_only(args)
    finally:
        for name, value in original.items():
            setattr(_metrics_runner, name, value)


def run_full_pipeline(args, defaults=None, cfg=None):
    sync = {
        "mne": mne,
        "np": np,
        "parse_token_map": parse_token_map,
        "brainvision_links_ok": brainvision_links_ok,
        "read_raw_preprocess": read_raw_preprocess,
        "compute_ica_diagnostics": compute_ica_diagnostics,
        "fit_ica": fit_ica,
        "find_ica_excludes": find_ica_excludes,
        "apply_ica": apply_ica,
        "events_from_annotations_positions": events_from_annotations_positions,
        "detect_trigger_bursts": detect_trigger_bursts,
        "read_eventcodes_from_subject_csv": read_eventcodes_from_subject_csv,
        "clean_eventcodes": clean_eventcodes,
        "filter_codes": filter_codes,
        "align_marker_positions_to_codes": align_marker_positions_to_codes,
        "build_events_from_positions_and_codes": build_events_from_positions_and_codes,
        "select_and_recode_stddev": select_and_recode_stddev,
        "select_and_filter_conditions": select_and_filter_conditions,
        "make_epochs": make_epochs,
        "derive_metadata_v1": derive_metadata_v1,
        "derive_metadata_from_condition_map": derive_metadata_from_condition_map,
        "moving_window_ptp_max": moving_window_ptp_max,
        "moving_window_ptp_mask": moving_window_ptp_mask,
        "simple_voltage_threshold_mask": simple_voltage_threshold_mask,
        "step_threshold_mask": step_threshold_mask,
        "compute_erp_metrics": compute_erp_metrics,
        "compute_erp_timeseries": compute_erp_timeseries,
        "compute_tfr_metrics": compute_tfr_metrics,
        "recommend_ica": recommend_ica,
        "compute_evokeds": compute_evokeds,
        "grand_averages": grand_averages,
        "write_qc_summary": write_qc_summary,
    }
    runner_original = {name: getattr(_pipeline_runner, name, _MISSING) for name in sync}
    metrics_original = {
        "build_erp_windows": _metrics_runner.build_erp_windows,
        "compute_erp_metrics": _metrics_runner.compute_erp_metrics,
        "compute_erp_timeseries": _metrics_runner.compute_erp_timeseries,
        "compute_tfr_metrics": _metrics_runner.compute_tfr_metrics,
    }
    try:
        for name, value in sync.items():
            setattr(_pipeline_runner, name, value)
        _metrics_runner.build_erp_windows = _build_erp_windows
        _metrics_runner.compute_erp_metrics = compute_erp_metrics
        _metrics_runner.compute_erp_timeseries = compute_erp_timeseries
        _metrics_runner.compute_tfr_metrics = compute_tfr_metrics
        return _pipeline_runner.run_full_pipeline(args, defaults=defaults, cfg=cfg)
    finally:
        for name, value in runner_original.items():
            if value is _MISSING:
                delattr(_pipeline_runner, name)
            else:
                setattr(_pipeline_runner, name, value)
        for name, value in metrics_original.items():
            setattr(_metrics_runner, name, value)


def summarize_one_file(args, raw_path: Path):
    sync = {
        "mne": mne,
        "read_raw_preprocess": read_raw_preprocess,
        "compute_ica_diagnostics": compute_ica_diagnostics,
        "events_from_annotations_positions": events_from_annotations_positions,
        "marker_gap_stats": marker_gap_stats,
        "keep_by_gap_heuristic": keep_by_gap_heuristic,
        "parse_vmrk_markers": parse_vmrk_markers,
        "read_eventcodes_from_subject_csv": read_eventcodes_from_subject_csv,
        "clean_eventcodes": clean_eventcodes,
        "filter_codes": filter_codes,
        "align_marker_positions_to_codes": align_marker_positions_to_codes,
        "parse_token_map": parse_token_map,
        "derive_metadata_v1": derive_metadata_v1,
    }
    original = {name: getattr(_summary_runner, name) for name in sync}
    try:
        for name, value in sync.items():
            setattr(_summary_runner, name, value)
        return _summary_runner.summarize_one_file(args, raw_path)
    finally:
        for name, value in original.items():
            setattr(_summary_runner, name, value)


def _prompt_yes_no(msg: str) -> bool:
    if not sys.stdin.isatty():
        return False
    resp = input(msg).strip().lower()
    return resp in {"y", "yes"}


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    ap.add_argument(
        "--erp-core",
        dest="erp_core",
        action="store_true",
        help="Use ERP CORE-style defaults (TP9/TP10, 0.1–20 Hz, ICA on, individualized thresholds).",
    )
    ap.add_argument("--process_data", action="store_true", help="Process raw data into epochs/evokeds/QC")
    ap.add_argument("--get_metrics", action="store_true", help="Compute ERP/TFR metrics")
    ap.add_argument("--plot_figures", action="store_true", help="Generate paper-ready figures")
    ap.add_argument("--raw_dir",  help="Folder containing BrainVision .vhdr or EEGLAB .set files (recurses)")
    ap.add_argument("--subject_csv_dir",  help="Folder containing subject-###.csv files")
    ap.add_argument("--out_dir", help="Output root folder")
    ap.add_argument("--summarize_one_file", default=None, help="If provided, summarize this raw file (.vhdr or .set) and exit.")

    ap.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration where available (MNE/CuPy).")
    ap.add_argument("--gpu_device", type=int, default=None, help="Optional GPU device index (default: first visible).")

    ap.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional list of subject stems to run (e.g., S203 s204). If omitted, runs all .vhdr/.set files in raw_dir.",
    )

    ap.add_argument(
        "--on_missing_subject_csv",
        choices=["skip", "fail"],
        default="skip",
        help="What to do if subject-###.csv is missing (default: skip).",
    )

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
        help="Keep only these EventCode values from subject-###.csv when aligning to EEG markers.",
    )
    ap.add_argument(
        "--eventcode_cleanup",
        default="none",
        choices=["none", "mprocacc_thesis"],
        help="Optional cleanup rule to apply to EventCode sequences before behavioral_keep_codes filtering.",
    )
    ap.add_argument(
        "--drop_eeg_markers_by_gap_s",
        type=float,
        default=None,
        help="Optional gap threshold heuristic (seconds) to drop likely boundary markers before auto-drop-to-count.",
    )
    ap.add_argument(
        "--collapse_eeg_marker_bursts_s",
        type=float,
        default=None,
        help="Optional ITI threshold (seconds) for collapsing bursty EEG markers before behavioral alignment.",
    )
    ap.add_argument(
        "--collapse_eeg_marker_bursts_keep",
        default="first",
        choices=["first", "last"],
        help="Which marker to keep from each collapsed burst (default: first).",
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
    ap.add_argument("--save_ica", default=1, type=int, help="Save ICA object to out_dir/00_ica (1=yes,0=no).")

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
        help="Compute ERP/TFR metrics and write to out_dir/05_metrics (1=yes,0=no).",
    )
    ap.add_argument(
        "--allow_metric_failures",
        action="store_true",
        help="Continue when subject-level metric computation fails. By default metric failures abort the run.",
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
    ap.add_argument("--figures_out_dir", default=None, help="Output directory for figures (default: out_dir/figures)")
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

    if not (args.process_data or args.get_metrics or args.plot_figures):
        # Default behavior: process data + metrics
        args.process_data = True
        args.get_metrics = True
        args.plot_figures = False

    # Apply ERP CORE preset before config so it can override config defaults.
    apply_erp_core_preset(args, defaults)

    # Apply config once for all stages
    cfg = apply_config(args, defaults)

    if args.summarize_one_file:
        summarize_one_file(args, Path(args.summarize_one_file))
        return

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

    if args.process_data:
        if not args.get_metrics:
            args.metrics = 0
        else:
            args.metrics = 1
        run_full_pipeline(args, defaults=defaults, cfg=cfg)
    elif args.get_metrics:
        run_metrics_only(args)

    if args.plot_figures:
        metrics_dir = Path(args.out_dir) / "05_metrics"
        erp_parquet = metrics_dir / "erp_timeseries_all.parquet"
        tfr_file = metrics_dir / "tfr_metrics_all.csv"

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
