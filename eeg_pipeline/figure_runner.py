from __future__ import annotations

from argparse import Namespace
from pathlib import Path


def run_plot_figures(args: Namespace) -> None:
    from .viz import paper_figures

    out_dir = Path(args.out_dir)
    metrics_dir = out_dir / "05_metrics"
    fig_dir = Path(args.figures_out_dir) if args.figures_out_dir else out_dir / "figures"

    erp_parquet = metrics_dir / "erp_timeseries_all.parquet"
    tfr_file = metrics_dir / "tfr_metrics_all.csv"

    erp_exists = erp_parquet.exists()
    tfr_exists = tfr_file.exists()

    if not erp_exists and not tfr_exists:
        raise FileNotFoundError(
            f"No metrics found for plotting. Expected {erp_parquet} and/or {tfr_file}."
        )

    time_window = resolve_figure_time_window(args)
    freq_band = resolve_figure_freq_band(args) if tfr_exists else None

    argv = [
        "--out_dir",
        str(fig_dir),
        "--time_window",
        str(time_window[0]),
        str(time_window[1]),
    ]
    if erp_exists:
        argv += ["--erp_parquet", str(erp_parquet)]
    if tfr_exists and freq_band is not None:
        argv += [
            "--tfr_file",
            str(tfr_file),
            "--freq_band",
            str(freq_band[0]),
            str(freq_band[1]),
        ]
    if args.figure_diff_heatmap:
        argv.append("--diff_heatmap")
    if args.figure_channels:
        argv += ["--channels", *args.figure_channels]

    paper_figures.main(argv)


def resolve_figure_time_window(args: Namespace) -> tuple[float, float]:
    if args.figure_time_window is not None:
        return float(args.figure_time_window[0]), float(args.figure_time_window[1])
    if getattr(args, "erp_window", None):
        window = args.erp_window[0]
        return float(window[1]), float(window[2])
    return float(args.tmin), float(args.tmax)


def resolve_figure_freq_band(args: Namespace) -> tuple[float, float] | None:
    if args.figure_freq_band is not None:
        return float(args.figure_freq_band[0]), float(args.figure_freq_band[1])
    return float(getattr(args, "tfr_fmin", 1.0)), float(getattr(args, "tfr_fmax", 30.0))
