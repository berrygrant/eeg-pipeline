"""run_analysis.py

Run ERP and/or time–frequency (TFR) metrics on pre-epoched EEG data.

This script aligns with eeg_pipeline.metrics APIs and supports:
- ERP windows by name (ERP_WINDOWS) or explicit NAME TMIN TMAX triples
- TFR metrics using tfr_tmin/tfr_tmax and tfr_fmin/fmax/fstep
- Optional aggregate labeling and figure outputs

Legacy compatibility:
- --tfr_window and --tfr_freqs are accepted as aliases for
  --tfr_tmin/--tfr_tmax and --tfr_fmin/--tfr_fmax/--tfr_fstep.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from eeg_pipeline.metrics import load_epochs, compute_erp_metrics, compute_tfr_metrics
from eeg_pipeline.metrics.tfr import TFRParams

# Optional: pre-defined ERP windows, matching run_metrics.py
try:
    from eeg_pipeline.metrics.erp_windows import ERP_WINDOWS  # type: ignore
except Exception:
    ERP_WINDOWS = {}  # type: ignore

try:
    from eeg_pipeline.metrics.erp_windows import ERPWindow  # type: ignore
except Exception:
    ERPWindow = None  # type: ignore


def _subject_from_filename(p: Path) -> str:
    # s203-epo.fif -> s203
    stem = p.stem
    if stem.endswith("-epo"):
        stem = stem[:-4]
    return stem


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run ERP and/or TFR metrics on previously epoched EEG data."
    )

    # I/O
    ap.add_argument("--epochs_dir", required=True, help="Folder containing *-epo.fif files")
    ap.add_argument("--out_dir", required=True, help="Output folder (e.g., 05_metrics)")
    ap.add_argument("--pattern", default="*-epo.fif", help="Glob pattern (default: *-epo.fif)")

    # Which analyses
    ap.add_argument("--do_erp", action="store_true", help="Run ERP metrics")
    ap.add_argument("--do_tfr", action="store_true", help="Run TFR metrics")

    # ERP settings (two ways: by name or by explicit window triple)
    ap.add_argument(
        "--erp_windows",
        nargs="+",
        default=None,
        help=(
            "ERP window names to compute (uses ERP_WINDOWS). "
            f"Available: {', '.join(ERP_WINDOWS.keys())}" if ERP_WINDOWS else
            "ERP window names to compute (uses ERP_WINDOWS)."
        ),
    )
    ap.add_argument(
        "--erp_window",
        nargs=3,
        metavar=("NAME", "TMIN", "TMAX"),
        action="append",
        default=[],
        help="Define an ERP analysis window by name and time bounds (sec). Repeatable.",
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
        action="store_true",
        help="Compute Deviant–Standard difference wave (MMN) if supported",
    )

    # TFR settings (defaults match run_metrics.py)
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

    # Legacy compatibility aliases
    ap.add_argument(
        "--tfr_window",
        nargs=2,
        type=float,
        metavar=("TMIN", "TMAX"),
        default=None,
        help="Deprecated alias for --tfr_tmin/--tfr_tmax.",
    )
    ap.add_argument(
        "--tfr_freqs",
        nargs=3,
        type=float,
        metavar=("FMIN", "FMAX", "FSTEP"),
        default=None,
        help="Deprecated alias for --tfr_fmin/--tfr_fmax/--tfr_fstep.",
    )

    # Output consistency / reporting
    ap.add_argument(
        "--aggregate_condition_label",
        default="ALL",
        help="Label to use when a row is aggregated across conditions (condition is blank/NA).",
    )
    ap.add_argument(
        "--split_aggregate_rows",
        action="store_true",
        help="If set, save aggregate-across-condition rows to a separate CSV instead of mixing them in.",
    )

    # Publication-ready outputs
    ap.add_argument(
        "--make_figures",
        action="store_true",
        help="If set, generate group-level figures (ERP and/or TFR) alongside CSVs.",
    )
    ap.add_argument(
        "--fig_format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Figure format for saved plots.",
    )
    ap.add_argument("--dpi", type=int, default=300, help="DPI for raster formats (png).")

    return ap


def _resolve_erp_windows(args: argparse.Namespace):
    # Priority:
    # 1) explicit --erp_window triples
    # 2) --erp_windows names (ERP_WINDOWS)
    windows = []

    if args.erp_window:
        if ERPWindow is None:
            raise RuntimeError(
                "You passed --erp_window triples, but ERPWindow could not be imported. "
                "Use --erp_windows <names> instead."
            )
        for name, tmin, tmax in args.erp_window:
            windows.append(ERPWindow(name=name, tmin=float(tmin), tmax=float(tmax)))

    if not windows:
        names = args.erp_windows or ["MMN"]
        if not ERP_WINDOWS:
            raise RuntimeError(
                "No ERP_WINDOWS mapping available, and you did not provide --erp_window triples. "
                "Either (a) pass --erp_window NAME TMIN TMAX, or (b) ensure ERP_WINDOWS is importable."
            )
        for name in names:
            if name not in ERP_WINDOWS:
                raise ValueError(
                    f"Unknown ERP window '{name}'. "
                    f"Available: {', '.join(ERP_WINDOWS.keys())}"
                )
            windows.append(ERP_WINDOWS[name])

    return windows


def _split_rows_with_missing_condition(df: pd.DataFrame):
    """Return (df_main, df_missing_cond) where missing condition means NA/empty string."""
    if "condition" not in df.columns:
        return df, None
    s = df["condition"]
    is_missing = s.isna() | (s.astype(str).str.strip() == "")
    df_main = df.loc[~is_missing].copy()
    df_missing = df.loc[is_missing].copy()
    return df_main, df_missing if len(df_missing) else None


def _label_missing_condition(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Replace missing/blank condition values with a human-readable label."""
    if df is None or "condition" not in df.columns:
        return df
    s = df["condition"]
    df["condition"] = s.replace("", pd.NA).fillna(label)
    return df


def _maybe_make_figures(
    *,
    out_dir: Path,
    erp_rows: List[pd.DataFrame],
    tfr_rows: List[pd.DataFrame],
    args: argparse.Namespace,
):
    if not args.make_figures:
        return

    # Figures are created from the raw data, not from the metrics frames.
    # To keep this script lightweight and dependency-safe, we only generate
    # group-level ERP/TFR plots if MNE and matplotlib are available.
    try:
        import mne  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:
        print(f"[WARN] --make_figures requested but plotting deps unavailable: {e}")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Re-load and grand-average for plots (keeps metrics path independent)
    epochs_dir = Path(args.epochs_dir)
    files = sorted(epochs_dir.glob(args.pattern))

    if args.do_erp:
        from mne import grand_average

        evokeds_by_cond = {c: [] for c in args.conditions}

        for p in files:
            loaded = load_epochs(p)
            for cond in args.conditions:
                if cond not in loaded.epochs.event_id:
                    continue
                ev = loaded.epochs[cond].average()
                ev.comment = f"{_subject_from_filename(p)}:{cond}"
                evokeds_by_cond[cond].append(ev)

        for cond, evs in evokeds_by_cond.items():
            if not evs:
                continue
            gav = grand_average(evs)
            fig = gav.plot(picks=args.channels, spatial_colors=False, show=False)
            fig_path = fig_dir / f"erp_{cond}.{args.fig_format}"
            fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass

            try:
                fig2 = gav.plot_joint(show=False)
                fig2_path = fig_dir / f"erp_joint_{cond}.{args.fig_format}"
                fig2.savefig(fig2_path, dpi=args.dpi, bbox_inches="tight")
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig2)
                except Exception:
                    pass
            except Exception as e:
                print(f"[WARN] Could not make joint ERP plot for {cond}: {e}")

    if args.do_tfr:
        import numpy as np
        import mne

        freqs = np.arange(args.tfr_fmin, args.tfr_fmax + 1e-9, args.tfr_fstep)

        def _compute_tfr(epochs):
            if args.tfr_method == "multitaper":
                return mne.time_frequency.tfr_multitaper(
                    epochs,
                    freqs=freqs,
                    n_cycles=freqs / args.tfr_n_cycles_div,
                    use_fft=True,
                    return_itc=False,
                    decim=args.tfr_decim,
                    average=True,
                )
            return mne.time_frequency.tfr_morlet(
                epochs,
                freqs=freqs,
                n_cycles=freqs / args.tfr_n_cycles_div,
                use_fft=True,
                return_itc=False,
                decim=args.tfr_decim,
                average=True,
            )

        for cond in args.conditions:
            tfrs = []
            for p in files:
                loaded = load_epochs(p)
                if cond not in loaded.epochs.event_id:
                    continue
                ep = loaded.epochs[cond]
                tfr = _compute_tfr(ep)
                if args.tfr_baseline is not None:
                    tfr.apply_baseline(tuple(args.tfr_baseline), mode=args.tfr_baseline_mode)
                tfrs.append(tfr)

            if not tfrs:
                continue

            data = np.mean([t.data for t in tfrs], axis=0)
            tfr_avg = tfrs[0].copy()
            tfr_avg.data = data
            tfr_avg.comment = f"GA:{cond}"

            try:
                fig = tfr_avg.plot(
                    picks=args.channels,
                    combine="mean",
                    show=False,
                    title=f"TFR (group avg) — {cond}",
                )
                fig_path = fig_dir / f"tfr_{cond}.{args.fig_format}"
                fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception:
                    pass
            except Exception as e:
                print(f"[WARN] Could not make TFR plot for {cond}: {e}")


def main(argv=None):
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    if not args.do_erp and not args.do_tfr:
        raise RuntimeError("No analysis selected. Use --do_erp and/or --do_tfr.")

    # Validate TFR baseline window lies within TFR time window
    if args.do_tfr and args.tfr_baseline is not None:
        b0, b1 = float(args.tfr_baseline[0]), float(args.tfr_baseline[1])
        if (b0 < args.tfr_tmin) or (b1 > args.tfr_tmax):
            raise ValueError(
                f"TFR baseline {args.tfr_baseline} must lie within tfr_tmin/tfr_tmax "
                f"({args.tfr_tmin}, {args.tfr_tmax})."
            )

    # Legacy alias mapping
    if args.tfr_window is not None:
        args.tfr_tmin, args.tfr_tmax = args.tfr_window
    if args.tfr_freqs is not None:
        args.tfr_fmin, args.tfr_fmax, args.tfr_fstep = args.tfr_freqs

    epochs_dir = Path(args.epochs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(epochs_dir.glob(args.pattern))
    if not files:
        raise RuntimeError(f"No files matched {args.pattern} in {epochs_dir}")

    windows = _resolve_erp_windows(args) if args.do_erp else None

    tfr_params = None
    if args.do_tfr:
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

        if args.do_erp:
            df_erp = compute_erp_metrics(
                epochs=loaded.epochs,
                subject=subj,
                channels=args.channels,
                windows=windows,
                conditions=args.conditions,
                compute_mmn=bool(args.compute_mmn),
            )
            erp_rows.append(df_erp)

        if args.do_tfr:
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

        msg = f"[OK] {subj}"
        if args.do_erp:
            msg += f" | ERP rows={len(erp_rows[-1])}"
        if args.do_tfr:
            msg += f" | TFR rows={len(tfr_rows[-1])}"
        print(msg)

    if args.do_erp:
        df_erp_all = pd.concat(erp_rows, ignore_index=True)

        if args.split_aggregate_rows:
            df_main, df_agg = _split_rows_with_missing_condition(df_erp_all)
            out_erp = out_dir / "erp_metrics.csv"
            df_main.to_csv(out_erp, index=False)
            print(f"\nSaved ERP metrics -> {out_erp}")

            if df_agg is not None and len(df_agg):
                df_agg = _label_missing_condition(df_agg, label=args.aggregate_condition_label)
                out_erp_agg = out_dir / "erp_metrics_aggregates.csv"
                df_agg.to_csv(out_erp_agg, index=False)
                print(f"Saved ERP aggregates -> {out_erp_agg}")
        else:
            df_erp_all = _label_missing_condition(df_erp_all, label=args.aggregate_condition_label)
            out_erp = out_dir / "erp_metrics.csv"
            df_erp_all.to_csv(out_erp, index=False)
            print(f"\nSaved ERP metrics -> {out_erp}")

    if args.do_tfr:
        df_tfr_all = pd.concat(tfr_rows, ignore_index=True)
        df_tfr_all = _label_missing_condition(df_tfr_all, label=args.aggregate_condition_label)
        out_tfr = out_dir / "tfr_metrics.csv"
        df_tfr_all.to_csv(out_tfr, index=False)
        print(f"Saved TFR metrics -> {out_tfr}")

    _maybe_make_figures(out_dir=out_dir, erp_rows=erp_rows, tfr_rows=tfr_rows, args=args)


if __name__ == "__main__":
    main()
