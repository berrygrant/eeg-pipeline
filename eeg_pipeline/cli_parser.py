from __future__ import annotations

import argparse


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
        "--fail_fast",
        action="store_true",
        help=(
            "Stop at the first recording that raises. By default the run continues "
            "and records an ERROR QC row for it, so one bad recording cannot hide "
            "the outcome of every other participant."
        ),
    )
    ap.add_argument(
        "--skip_aggregate",
        action="store_true",
        help=(
            "Process subjects but do not rebuild dataset-level outputs. Required when "
            "running subjects concurrently (e.g. a SLURM array), where each task must "
            "write only its own derivatives and a single later --aggregate_only job "
            "combines them. Without this, concurrent tasks race on the shared tables."
        ),
    )
    ap.add_argument(
        "--aggregate_only",
        action="store_true",
        help=(
            "Rebuild dataset-level QC/metrics tables and grand averages from existing "
            "per-subject derivatives, without reprocessing. Use as the gather step after "
            "running subjects independently (e.g. one SLURM array task per subject)."
        ),
    )
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
        "--n_jobs",
        type=int,
        # Sentinel, not 1: `provided()` detects an explicit flag by comparing
        # against the argparse default, so a default of 1 would make an explicit
        # `--n_jobs 1` indistinguishable from omitting it -- and unable to
        # override a larger compute.n_jobs in the config. That matters because
        # hpc/slurm_array.sbatch passes exactly 1 when SLURM_CPUS_PER_TASK is
        # unset, where silently using the config's value would oversubscribe.
        default=None,
        help=(
            "Worker processes for MNE operations that parallelize across channels "
            "(filtering, notch, TFR). -1 uses all cores. On a cluster keep this equal "
            "to --cpus-per-task, and set OMP_NUM_THREADS to match, or threaded BLAS "
            "will oversubscribe the allocation."
        ),
    )

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


