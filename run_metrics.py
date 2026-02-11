# run_metrics.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eeg_pipeline.metrics import (
    load_epochs,
    compute_erp_metrics,
    compute_tfr_metrics,
)
from eeg_pipeline.metrics.erp_windows import ERP_WINDOWS
from eeg_pipeline.metrics.tfr import TFRParams


def _subject_from_filename(p: Path) -> str:
    # s203-epo.fif -> s203
    stem = p.stem
    if stem.endswith("-epo"):
        stem = stem[:-4]
    return stem


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Run ERP and TFR metrics on previously epoched EEG data."
    )

    # I/O
    ap.add_argument("--epochs_dir", required=True, help="Folder containing *-epo.fif files")
    ap.add_argument("--out_dir", required=True, help="Output folder (e.g., 05_metrics)")
    ap.add_argument("--pattern", default="*-epo.fif", help="Glob pattern (default: *-epo.fif)")

    # ERP settings
    ap.add_argument(
        "--erp_windows",
        nargs="+",
        default=["MMN"],
        help=f"ERP window names to compute. Available: {', '.join(ERP_WINDOWS.keys())}",
    )
    ap.add_argument(
        "--channels",
        nargs="+",
        default=["Fp1", "Fz", "Cz"],
        help="Channels to analyze",
    )
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=["Standard", "Deviant"],
        help="Condition names in epochs.event_id",
    )
    ap.add_argument(
        "--compute_mmn",
        type=int,
        default=1,
        help="Compute Deviant–Standard difference wave (1=yes, 0=no)",
    )

    # TFR settings
    ap.add_argument("--tfr_tmin", type=float, default=-0.2)
    ap.add_argument("--tfr_tmax", type=float, default=0.6)
    ap.add_argument("--tfr_fmin", type=float, default=1.0)
    ap.add_argument("--tfr_fmax", type=float, default=30.0)
    ap.add_argument("--tfr_fstep", type=float, default=1.0)
    ap.add_argument(
        "--tfr_method",
        default="multitaper",
        choices=["multitaper", "morlet"],
    )
    ap.add_argument("--tfr_n_cycles_div", type=float, default=10.0)
    ap.add_argument("--tfr_decim", type=int, default=1)
    ap.add_argument(
        "--tfr_baseline",
        nargs=2,
        type=float,
        default=[-0.1, 0.0],
        metavar=("TMIN", "TMAX"),
    )
    ap.add_argument("--tfr_baseline_mode", default="logratio")

    return ap


def main(argv=None):
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    epochs_dir = Path(args.epochs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(epochs_dir.glob(args.pattern))
    if not files:
        raise RuntimeError(f"No files matched {args.pattern} in {epochs_dir}")

    # Resolve ERP windows
    windows = []
    for name in args.erp_windows:
        if name not in ERP_WINDOWS:
            raise ValueError(
                f"Unknown ERP window '{name}'. "
                f"Available: {', '.join(ERP_WINDOWS.keys())}"
            )
        windows.append(ERP_WINDOWS[name])

    # TFR parameters
    tfr_params = TFRParams(
        fmin=args.tfr_fmin,
        fmax=args.tfr_fmax,
        fstep=args.tfr_fstep,
        method=args.tfr_method,
        n_cycles_div=args.tfr_n_cycles_div,
        decim=args.tfr_decim,
        baseline=(args.tfr_baseline[0], args.tfr_baseline[1]),
        mode=args.tfr_baseline_mode,
    )

    erp_rows = []
    tfr_rows = []

    for p in files:
        subj = _subject_from_filename(p)
        loaded = load_epochs(p)

        # ---- ERP metrics ----
        df_erp = compute_erp_metrics(
            epochs=loaded.epochs,
            subject=subj,
            channels=args.channels,
            windows=windows,
            conditions=args.conditions,
        )
        erp_rows.append(df_erp)

        # ---- TFR metrics ----
        df_tfr = compute_tfr_metrics(
            epochs=loaded.epochs,
            subject=subj,
            channels=args.channels,
            conditions=args.conditions,
            tmin=args.tfr_tmin,
            tmax=args.tfr_tmax,
            params=tfr_params,
        )
        tfr_rows.append(df_tfr)

        print(
            f"[OK] {subj}: "
            f"ERP rows={len(df_erp)} | "
            f"TFR rows={len(df_tfr)}"
        )

    # Concatenate + save
    df_erp_all = pd.concat(erp_rows, ignore_index=True)
    df_tfr_all = pd.concat(tfr_rows, ignore_index=True)

    out_erp = out_dir / "erp_metrics.csv"
    out_tfr = out_dir / "tfr_metrics.csv"

    df_erp_all.to_csv(out_erp, index=False)
    df_tfr_all.to_csv(out_tfr, index=False)

    print(f"\nSaved ERP metrics -> {out_erp}")
    print(f"Saved TFR metrics -> {out_tfr}")


if __name__ == "__main__":
    main()
