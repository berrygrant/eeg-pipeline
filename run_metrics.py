# run_metrics.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eeg_pipeline.metrics import load_epochs, compute_erp_metrics, compute_tfr_metrics
from eeg_pipeline.metrics.erp import ERPWindow
from eeg_pipeline.metrics.tfr import TFRParams
from eeg_pipeline.metrics.erp_timeseries import compute_erp_timeseries, ERPTimeSeriesParams


def _subject_from_filename(p: Path) -> str:
    # e.g., s203-epo.fif -> s203 ; s203.set -> s203
    stem = p.stem
    if stem.endswith("-epo"):
        stem = stem[:-4]
    return stem


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Run ERP + TFR metrics on epoched data.")
    ap.add_argument("--epochs_dir", required=True, help="Folder with *-epo.fif OR *.set")
    ap.add_argument("--out_dir", required=True, help="Output folder for metrics CSVs")

    ap.add_argument("--pattern", default="*-epo.fif", help="Glob pattern (default: *-epo.fif). Use *.set for EEGLAB.")
    ap.add_argument("--channels", nargs="+", default=["Fp1", "Fz", "F3", "Cz", "F4"], help="Channels to analyze")
    ap.add_argument("--conditions", nargs="+", default=["Standard", "Deviant"], help="Condition names in epochs.event_id")

    # ERP windows
    ap.add_argument("--erp_window", nargs=3, action="append", metavar=("NAME", "TMIN", "TMAX"),
                    help="Add an ERP window, e.g. --erp_window MMN_150_250 0.15 0.25. Can be repeated.")
    ap.add_argument("--compute_mmn", type=int, default=1, help="Compute MMN = Deviant-Standard (1=yes,0=no)")

    # TFR settings
    ap.add_argument("--tfr_tmin", type=float, default=-0.2)
    ap.add_argument("--tfr_tmax", type=float, default=0.6)
    ap.add_argument("--tfr_fmin", type=float, default=1.0)
    ap.add_argument("--tfr_fmax", type=float, default=30.0)
    ap.add_argument("--tfr_fstep", type=float, default=1.0)
    ap.add_argument("--tfr_method", default="multitaper", choices=["multitaper", "morlet"])
    ap.add_argument("--tfr_n_cycles_div", type=float, default=10.0)
    ap.add_argument("--tfr_decim", type=int, default=1)
    ap.add_argument("--tfr_baseline", nargs=2, type=float, default=[-0.1, 0.0])
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

    # ERP windows
    if args.erp_window:
        windows = [ERPWindow(name=w[0], tmin=float(w[1]), tmax=float(w[2])) for w in args.erp_window]
    else:
        # sensible default for MMN-style component window
        windows = [ERPWindow("MMN_150_250", 0.15, 0.25)]

    tfr_params = TFRParams(
        fmin=args.tfr_fmin,
        fmax=args.tfr_fmax,
        fstep=args.tfr_fstep,
        method=args.tfr_method,
        n_cycles_div=args.tfr_n_cycles_div,
        decim=args.tfr_decim,
        baseline=(float(args.tfr_baseline[0]), float(args.tfr_baseline[1])),
        mode=args.tfr_baseline_mode,
    )

    params = ERPTimeSeriesParams(
        tmin=args.erp_ts_tmin,
        tmax=args.erp_ts_tmax,
        baseline=(args.baseline_tmin, args.baseline_tmax),
        decim=args.erp_ts_decim,
    )

df_ts.to_parquet(out_dir / "erp_timeseries.parquet", index=False)

    erp_all = []
    tfr_all = []

    for p in files:
        subj = _subject_from_filename(p)
        loaded = load_epochs(p)

        # NOTE: if your epochs are empty, these will still emit status rows rather than crash.
        df_erp = compute_erp_metrics(
            loaded.epochs,
            subject=subj,
            channels=args.channels,
            windows=windows,
            conditions=args.conditions,
            compute_mmn=bool(args.compute_mmn),
        )
        df_tfr = compute_tfr_metrics(
            loaded.epochs,
            subject=subj,
            channels=args.channels,
            tmin=args.tfr_tmin,
            tmax=args.tfr_tmax,
            conditions=args.conditions,
            params=tfr_params,
        )
        df_ts = compute_erp_timeseries(
            epochs,
            subject=subj,
            channels=args.channels,
            params=params,
            include_difference_wave=bool(args.erp_ts_include_diff),
        )
        
        erp_all.append(df_erp)
        tfr_all.append(df_tfr)


        print(f"[OK] {subj}: ERP rows={len(df_erp)} | TFR rows={len(df_tfr)}")

    df_erp_all = pd.concat(erp_all, ignore_index=True)
    df_tfr_all = pd.concat(tfr_all, ignore_index=True)

    out_erp = out_dir / "erp_metrics.csv"
    out_tfr = out_dir / "tfr_metrics.csv"
    df_erp_all.to_csv(out_erp, index=False)
    df_tfr_all.to_csv(out_tfr, index=False)

    print(f"\nSaved -> {out_erp}")
    print(f"Saved -> {out_tfr}")


if __name__ == "__main__":
    main()