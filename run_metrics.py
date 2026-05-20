from __future__ import annotations

import argparse

from eeg_pipeline.analysis_runner import main as analysis_main
from eeg_pipeline.metrics.erp_windows import ERP_WINDOWS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ERP and TFR metrics on previously epoched EEG data."
    )
    parser.add_argument("--epochs_dir", required=True, help="Folder containing *-epo.fif files")
    parser.add_argument("--out_dir", required=True, help="Output folder (e.g., 05_metrics)")
    parser.add_argument("--pattern", default="*-epo.fif", help="Glob pattern (default: *-epo.fif)")
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration where available (MNE/CuPy).")
    parser.add_argument("--gpu_device", type=int, default=None, help="Optional GPU device index (default: first visible).")
    parser.add_argument(
        "--erp_windows",
        nargs="+",
        default=["MMN"],
        help=f"ERP window names to compute. Available: {', '.join(ERP_WINDOWS.keys())}",
    )
    parser.add_argument("--channels", nargs="+", default=["Fp1", "Fz", "Cz"], help="Channels to analyze")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["Standard", "Deviant"],
        help="Condition names in epochs.event_id",
    )
    parser.add_argument(
        "--compute_mmn",
        type=int,
        default=1,
        help="Compute Deviant-Standard difference wave (1=yes, 0=no)",
    )
    parser.add_argument(
        "--difference_label",
        default=None,
        help="Optional label for the Deviant-Standard difference wave (default: DEV_MINUS_STD).",
    )
    parser.add_argument("--tfr_tmin", type=float, default=-0.2)
    parser.add_argument("--tfr_tmax", type=float, default=0.6)
    parser.add_argument("--tfr_fmin", type=float, default=1.0)
    parser.add_argument("--tfr_fmax", type=float, default=30.0)
    parser.add_argument("--tfr_fstep", type=float, default=1.0)
    parser.add_argument("--tfr_method", default="multitaper", choices=["multitaper", "morlet"])
    parser.add_argument("--tfr_n_cycles_div", type=float, default=10.0)
    parser.add_argument("--tfr_decim", type=int, default=1)
    parser.add_argument("--tfr_baseline", nargs=2, type=float, default=[-0.1, 0.0], metavar=("TMIN", "TMAX"))
    parser.add_argument("--tfr_baseline_mode", default="logratio")
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    analysis_argv = [
        "--epochs_dir",
        args.epochs_dir,
        "--out_dir",
        args.out_dir,
        "--pattern",
        args.pattern,
        "--do_erp",
        "--do_tfr",
        "--channels",
        *args.channels,
        "--conditions",
        *args.conditions,
        "--erp_windows",
        *args.erp_windows,
        "--tfr_tmin",
        str(args.tfr_tmin),
        "--tfr_tmax",
        str(args.tfr_tmax),
        "--tfr_fmin",
        str(args.tfr_fmin),
        "--tfr_fmax",
        str(args.tfr_fmax),
        "--tfr_fstep",
        str(args.tfr_fstep),
        "--tfr_method",
        args.tfr_method,
        "--tfr_n_cycles_div",
        str(args.tfr_n_cycles_div),
        "--tfr_decim",
        str(args.tfr_decim),
        "--tfr_baseline",
        str(args.tfr_baseline[0]),
        str(args.tfr_baseline[1]),
        "--tfr_baseline_mode",
        args.tfr_baseline_mode,
    ]
    if args.use_gpu:
        analysis_argv.append("--use_gpu")
    if args.gpu_device is not None:
        analysis_argv.extend(["--gpu_device", str(args.gpu_device)])
    if bool(args.compute_mmn):
        analysis_argv.append("--compute_mmn")
    if args.difference_label is not None:
        analysis_argv.extend(["--difference_label", args.difference_label])

    analysis_main(analysis_argv)


if __name__ == "__main__":
    main()
