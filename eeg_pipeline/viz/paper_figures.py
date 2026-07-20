"""Generate paper-ready ERP and TFR figures from metrics outputs.

Inputs
- ERP time-series Parquet (from eeg_pipeline.metrics.erp_timeseries)
- TFR metrics CSV/Parquet (from eeg_pipeline.metrics.tfr.compute_tfr_metrics)

Figures
- ERP grand average across electrodes (Standard vs Deviant)
- ERP grand average per electrode (Standard vs Deviant)
- TFR evoked-power time series (avg over channels+freq band)
- TFR ITC time series (avg over channels+freq band)
- TFR evoked-power heatmap (freq x time)
- Half-violin distributions of evoked/induced power and ITC in a time window
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, cols: Iterable[str], *, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _filter_conditions(df: pd.DataFrame, *, standard: str, deviant: str) -> pd.DataFrame:
    return df[df["condition"].isin([standard, deviant])].copy()


def _save_fig(fig, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")


def _erp_grand_average(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    standard: str,
    deviant: str,
    time_window: tuple[float, float],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    tmin, tmax = time_window
    df = _filter_conditions(df, standard=standard, deviant=deviant)
    df = df[(df["time_s"] >= tmin) & (df["time_s"] <= tmax)]

    # Average over channels within subject, then across subjects
    subj_avg = (
        df.groupby(["subject", "condition", "time_s"], as_index=False)["amplitude_uv"]
        .mean()
    )
    ga = (
        subj_avg.groupby(["condition", "time_s"], as_index=False)["amplitude_uv"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    for cond, color in [(standard, "black"), (deviant, "red")]:
        s = ga[ga["condition"] == cond]
        ax.plot(s["time_s"], s["amplitude_uv"], color=color, label=cond)

    ax.axvspan(tmin, tmax, color="gray", alpha=0.2, zorder=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (uV)")
    ax.set_title("ERP grand average (all electrodes)")
    ax.legend(frameon=False)

    _save_fig(fig, out_dir / "erp_grand_average.png", dpi)
    plt.close(fig)


def _erp_by_channel(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    standard: str,
    deviant: str,
    time_window: tuple[float, float],
    dpi: int,
    channels: list[str] | None,
) -> None:
    import matplotlib.pyplot as plt

    tmin, tmax = time_window
    df = _filter_conditions(df, standard=standard, deviant=deviant)
    df = df[(df["time_s"] >= tmin) & (df["time_s"] <= tmax)]

    if channels:
        df = df[df["channel"].isin(channels)]

    # Average across subjects per channel
    ga = (
        df.groupby(["channel", "condition", "time_s"], as_index=False)["amplitude_uv"]
        .mean()
    )

    chs = sorted(ga["channel"].unique())
    if not chs:
        raise ValueError("No channels available after filtering.")

    n_cols = 4 if len(chs) > 8 else 3
    n_rows = int(np.ceil(len(chs) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i, ch in enumerate(chs):
        ax = axes[i]
        for cond, color in [(standard, "black"), (deviant, "red")]:
            s = ga[(ga["channel"] == ch) & (ga["condition"] == cond)]
            ax.plot(s["time_s"], s["amplitude_uv"], color=color)
        ax.axvspan(tmin, tmax, color="gray", alpha=0.2, zorder=0)
        ax.set_title(ch)

    for j in range(len(chs), len(axes)):
        axes[j].axis("off")

    fig.supxlabel("Time (s)")
    fig.supylabel("Amplitude (uV)")
    fig.suptitle("ERP grand average by electrode")

    _save_fig(fig, out_dir / "erp_by_channel.png", dpi)
    plt.close(fig)


def _tfr_timeseries(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    standard: str,
    deviant: str,
    time_window: tuple[float, float],
    freq_band: tuple[float, float],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    tmin, tmax = time_window
    fmin, fmax = freq_band

    df = _filter_conditions(df, standard=standard, deviant=deviant)
    df = df[(df["frequency"] >= fmin) & (df["frequency"] <= fmax)]
    df = df[(df["time"] >= tmin) & (df["time"] <= tmax)]

    # Average over channels+freq within subject, then across subjects
    subj_avg = (
        df.groupby(["subject", "condition", "time"], as_index=False)[["evoked_power", "itc"]]
        .mean()
    )
    ga = (
        subj_avg.groupby(["condition", "time"], as_index=False)[["evoked_power", "itc"]]
        .mean()
    )
    # ERPLAB-style: apply log10 transform after averaging
    eps = 1e-30
    ga["evoked_power_log10"] = np.log10(np.clip(ga["evoked_power"].to_numpy(), eps, None))

    # Evoked power time series
    fig, ax = plt.subplots(figsize=(7, 4))
    for cond, color in [(standard, "black"), (deviant, "red")]:
        s = ga[ga["condition"] == cond]
        ax.plot(s["time"], s["evoked_power_log10"], color=color, label=cond)
    ax.axvspan(tmin, tmax, color="gray", alpha=0.2, zorder=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Evoked power (log10)")
    ax.set_title("TFR evoked power (avg over freq+channels)")
    ax.legend(frameon=False)
    _save_fig(fig, out_dir / "tfr_evoked_timeseries.png", dpi)
    plt.close(fig)

    # ITC time series
    fig, ax = plt.subplots(figsize=(7, 4))
    for cond, color in [(standard, "black"), (deviant, "red")]:
        s = ga[ga["condition"] == cond]
        if s["itc"].isna().all():
            print(f"[WARN] ITC is all NaN for condition '{cond}' after filtering.")
        ax.plot(s["time"], s["itc"], color=color, label=cond)
    ax.axvspan(tmin, tmax, color="gray", alpha=0.2, zorder=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ITC")
    ax.set_title("TFR ITC (avg over freq+channels)")
    ax.legend(frameon=False)
    _save_fig(fig, out_dir / "tfr_itc_timeseries.png", dpi)
    plt.close(fig)


def _tfr_evoked_heatmap(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    standard: str,
    deviant: str,
    time_window: tuple[float, float],
    freq_band: tuple[float, float],
    dpi: int,
    diff_heatmap: bool,
) -> None:
    import matplotlib.pyplot as plt

    tmin, tmax = time_window
    fmin, fmax = freq_band

    df = _filter_conditions(df, standard=standard, deviant=deviant)
    df = df[(df["frequency"] >= fmin) & (df["frequency"] <= fmax)]
    df = df[(df["time"] >= tmin) & (df["time"] <= tmax)]

    # Average across subjects+channels
    avg = (
        df.groupby(["condition", "frequency", "time"], as_index=False)["evoked_power"]
        .mean()
    )
    # ERPLAB-style: apply log10 transform after averaging
    eps = 1e-30
    avg["evoked_power_log10"] = np.log10(np.clip(avg["evoked_power"].to_numpy(), eps, None))

    conds = [c for c in [standard, deviant] if c in avg["condition"].unique()]
    if not conds:
        return

    n_cols = len(conds)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4), sharey=True)
    axes = np.atleast_1d(axes)

    # Shared color scale across conditions
    pivots = []
    for cond in conds:
        sub = avg[avg["condition"] == cond]
        pivot = sub.pivot_table(index="frequency", columns="time", values="evoked_power_log10")
        pivots.append(pivot)
    all_vals = np.concatenate([p.values.ravel() for p in pivots]) if pivots else np.array([])
    if all_vals.size:
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
    else:
        vmin, vmax = None, None

    for ax, cond in zip(axes, conds, strict=False):
        sub = avg[avg["condition"] == cond]
        pivot = sub.pivot_table(index="frequency", columns="time", values="evoked_power_log10")
        times = pivot.columns.values
        freqs = pivot.index.values
        im = ax.pcolormesh(times, freqs, pivot.values, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"Evoked power heatmap ({cond}, log10)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=ax)

    _save_fig(fig, out_dir / "tfr_evoked_heatmap.png", dpi)
    plt.close(fig)

    if diff_heatmap and (standard in conds) and (deviant in conds):
        std = avg[avg["condition"] == standard]
        dev = avg[avg["condition"] == deviant]

        std_p = std.pivot_table(index="frequency", columns="time", values="evoked_power_log10")
        dev_p = dev.pivot_table(index="frequency", columns="time", values="evoked_power_log10")

        # Align on common time/freq grid
        freqs = std_p.index.intersection(dev_p.index)
        times = std_p.columns.intersection(dev_p.columns)
        if len(freqs) == 0 or len(times) == 0:
            return

        diff = dev_p.loc[freqs, times] - std_p.loc[freqs, times]

        fig, ax = plt.subplots(figsize=(6, 4))
        vmax = float(np.nanmax(np.abs(diff.values)))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        im = ax.pcolormesh(
            times.values,
            freqs.values,
            diff.values,
            shading="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title("Evoked power heatmap (Deviant - Standard, log10)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=ax)

        _save_fig(fig, out_dir / "tfr_evoked_heatmap_diff.png", dpi)
        plt.close(fig)


def _tfr_half_violin(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    standard: str,
    deviant: str,
    time_window: tuple[float, float],
    freq_band: tuple[float, float],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    tmin, tmax = time_window
    fmin, fmax = freq_band

    df = _filter_conditions(df, standard=standard, deviant=deviant)
    df = df[(df["frequency"] >= fmin) & (df["frequency"] <= fmax)]
    df = df[(df["time"] >= tmin) & (df["time"] <= tmax)]

    # Subject-level summary within time/freq window
    agg = (
        df.groupby(["subject", "condition"], as_index=False)[
            ["evoked_power", "induced_power", "itc"]
        ]
        .mean()
    )
    # ERPLAB-style: apply log10 transform after averaging
    eps = 1e-30
    agg["log10_evoked"] = np.log10(np.clip(agg["evoked_power"].to_numpy(), eps, None))
    agg["log10_induced"] = np.log10(np.clip(agg["induced_power"].to_numpy(), eps, None))

    long_df = agg.melt(
        id_vars=["subject", "condition"],
        value_vars=["log10_evoked", "log10_induced", "itc"],
        var_name="metric",
        value_name="value",
    )
    long_df["metric"] = long_df["metric"].map(
        {
            "log10_evoked": "Evoked Power (log10)",
            "log10_induced": "Induced Power (log10)",
            "itc": "ITC",
        }
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    conds = [c for c in [standard, deviant] if c in long_df["condition"].unique()]
    palette = {standard: "black", deviant: "red"}

    if len(conds) == 2:
        sns.violinplot(
            data=long_df,
            x="metric",
            y="value",
            hue="condition",
            hue_order=conds,
            split=True,
            inner="quartile",
            palette=palette,
            ax=ax,
        )
    else:
        color = palette.get(conds[0], "gray") if conds else "gray"
        sns.violinplot(
            data=long_df,
            x="metric",
            y="value",
            color=color,
            inner="quartile",
            ax=ax,
        )
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    ax.set_title("TFR distributions in window")
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    if ax.get_legend() is not None:
        ax.legend(frameon=False, title="")

    _save_fig(fig, out_dir / "tfr_violin.png", dpi)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate ERP/TFR figures from metrics outputs.")

    ap.add_argument("--erp_parquet", type=str, default=None, help="Path to ERP time-series parquet")
    ap.add_argument("--tfr_file", type=str, default=None, help="Path to TFR metrics (csv or parquet)")
    ap.add_argument("--out_dir", required=True, help="Output directory for figures")

    ap.add_argument("--standard_label", default="Standard")
    ap.add_argument("--deviant_label", default="Deviant")

    ap.add_argument("--time_window", nargs=2, type=float, required=True, metavar=("TMIN", "TMAX"))
    ap.add_argument("--freq_band", nargs=2, type=float, default=None, metavar=("FMIN", "FMAX"))
    ap.add_argument("--diff_heatmap", action="store_true", help="Also save deviant-standard heatmap.")

    ap.add_argument("--channels", nargs="+", default=None, help="Optional channel subset for ERP plots")
    ap.add_argument("--dpi", type=int, default=300)

    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    tmin, tmax = float(args.time_window[0]), float(args.time_window[1])
    time_window = (tmin, tmax)

    if args.erp_parquet:
        erp_path = Path(args.erp_parquet)
        df_erp = _read_table(erp_path)
        _require_columns(
            df_erp,
            ["subject", "condition", "channel", "time_s", "amplitude_uv"],
            label="ERP parquet",
        )
        if "status" in df_erp.columns:
            df_erp = df_erp[df_erp["status"] == "OK"]

        _erp_grand_average(
            df_erp,
            out_dir=out_dir,
            standard=args.standard_label,
            deviant=args.deviant_label,
            time_window=time_window,
            dpi=args.dpi,
        )
        _erp_by_channel(
            df_erp,
            out_dir=out_dir,
            standard=args.standard_label,
            deviant=args.deviant_label,
            time_window=time_window,
            dpi=args.dpi,
            channels=args.channels,
        )

    if args.tfr_file:
        if args.freq_band is None:
            raise ValueError("--freq_band is required when --tfr_file is provided")

        tfr_path = Path(args.tfr_file)
        df_tfr = _read_table(tfr_path)
        _require_columns(
            df_tfr,
            ["subject", "condition", "channel", "frequency", "time", "evoked_power", "induced_power", "itc"],
            label="TFR metrics",
        )
        if "status" in df_tfr.columns:
            df_tfr = df_tfr[df_tfr["status"] == "OK"]

        freq_band = (float(args.freq_band[0]), float(args.freq_band[1]))

        _tfr_timeseries(
            df_tfr,
            out_dir=out_dir,
            standard=args.standard_label,
            deviant=args.deviant_label,
            time_window=time_window,
            freq_band=freq_band,
            dpi=args.dpi,
        )
        _tfr_evoked_heatmap(
            df_tfr,
            out_dir=out_dir,
            standard=args.standard_label,
            deviant=args.deviant_label,
            time_window=time_window,
            freq_band=freq_band,
            dpi=args.dpi,
            diff_heatmap=bool(args.diff_heatmap),
        )
        _tfr_half_violin(
            df_tfr,
            out_dir=out_dir,
            standard=args.standard_label,
            deviant=args.deviant_label,
            time_window=time_window,
            freq_band=freq_band,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
