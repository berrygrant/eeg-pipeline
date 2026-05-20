from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import mne

from .metrics import compute_erp_metrics, compute_tfr_metrics, load_epochs
from .metrics.erp import ERPWindow
from .metrics.erp_timeseries import ERPTimeSeriesParams, compute_erp_timeseries
from .metrics.erp_windows import ERP_WINDOWS
from .metrics.tfr import TFRParams
from .metrics.writers import (
    ParquetRowGroupWriter,
    append_csv,
    reset_combined_metric_outputs,
)


def subject_from_epochs_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("-epo"):
        stem = stem[:-4]
    return stem


def resolve_metrics_conditions(args: Namespace) -> list[str]:
    metrics_conditions = getattr(args, "metrics_conditions", None)
    if metrics_conditions:
        return list(metrics_conditions)

    cond_map = getattr(args, "condition_map", None)
    if cond_map:
        return list(cond_map.keys())

    return ["Standard", "Deviant"]


def build_erp_windows(args: Namespace) -> list[ERPWindow]:
    if getattr(args, "erp_window", None):
        return [
            ERPWindow(name=w[0], tmin=float(w[1]), tmax=float(w[2]))
            for w in args.erp_window
        ]

    windows: list[ERPWindow] = []
    if bool(getattr(args, "compute_mmn", 0)):
        windows.append(ERP_WINDOWS["MMN"])
    if bool(getattr(args, "compute_p300", 0)):
        windows.append(ERP_WINDOWS["P300"])
    return windows


def build_tfr_params(args: Namespace) -> TFRParams:
    baseline = getattr(args, "tfr_baseline", [-0.1, 0.0])
    return TFRParams(
        fmin=float(getattr(args, "tfr_fmin", 1.0)),
        fmax=float(getattr(args, "tfr_fmax", 30.0)),
        fstep=float(getattr(args, "tfr_fstep", 1.0)),
        method=str(getattr(args, "tfr_method", "multitaper")),
        n_cycles_div=float(getattr(args, "tfr_n_cycles_div", 10.0)),
        decim=int(getattr(args, "tfr_decim", 1)),
        baseline=(float(baseline[0]), float(baseline[1])),
        mode=str(getattr(args, "tfr_baseline_mode", "logratio")),
    )


def write_subject_metrics(
    *,
    epochs: mne.Epochs,
    subject: str,
    args: Namespace,
    metrics_dir: Path,
    conditions: list[str] | None = None,
    parquet_writer: ParquetRowGroupWriter | None = None,
) -> dict[str, int]:
    do_erp = bool(getattr(args, "metrics_erp_enabled", True))
    do_tfr = bool(getattr(args, "metrics_tfr_enabled", True))
    if not (do_erp or do_tfr or bool(getattr(args, "metrics_erp_timeseries", False))):
        return {}

    metrics_dir.mkdir(parents=True, exist_ok=True)
    channels = getattr(args, "metrics_channels", None) or ["Fp1", "Fz", "Cz"]
    conditions = conditions or resolve_metrics_conditions(args)
    counts: dict[str, int] = {}

    if do_erp:
        try:
            diff_label = getattr(args, "difference_label", None)
            df_erp = compute_erp_metrics(
                epochs,
                subject=subject,
                channels=channels,
                conditions=conditions,
                windows=build_erp_windows(args),
                compute_mmn=bool(getattr(args, "compute_mmn", 1)),
                mmn_name=diff_label if diff_label else "DEV_MINUS_STD",
            )
            df_erp.to_csv(metrics_dir / f"{subject}_erp_metrics.csv", index=False)
            append_csv(df_erp, metrics_dir / "erp_metrics_all.csv")
            counts["erp_rows"] = len(df_erp)
        except Exception as e:
            _handle_metric_failure(args, f"ERP metrics failed for {subject}: {e}", e)

    if bool(getattr(args, "metrics_erp_timeseries", False)):
        try:
            if getattr(args, "metrics_channels", None) is None:
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
                subject=subject,
                channels=ts_channels,
                params=ts_params,
                conditions=conditions,
                include_difference_wave=False,
            )
            ts_dir = metrics_dir / "erp_timeseries"
            ts_dir.mkdir(parents=True, exist_ok=True)
            df_ts.to_parquet(ts_dir / f"{subject}_erp_timeseries.parquet", index=False)
            if parquet_writer is not None:
                parquet_writer.write(df_ts, metrics_dir / "erp_timeseries_all.parquet")
            counts["erp_timeseries_rows"] = len(df_ts)
        except Exception as e:
            _handle_metric_failure(args, f"ERP timeseries failed for {subject}: {e}", e)

    if do_tfr:
        try:
            df_tfr = compute_tfr_metrics(
                epochs,
                subject=subject,
                channels=channels,
                conditions=conditions,
                params=build_tfr_params(args),
                tmin=float(getattr(args, "tfr_tmin", -0.2)),
                tmax=float(getattr(args, "tfr_tmax", 0.6)),
                time_decim=int(getattr(args, "tfr_time_decim", 1)),
            )
            df_tfr.to_csv(metrics_dir / f"{subject}_tfr_metrics.csv", index=False)
            append_csv(df_tfr, metrics_dir / "tfr_metrics_all.csv")
            counts["tfr_rows"] = len(df_tfr)
        except Exception as e:
            _handle_metric_failure(args, f"TFR metrics failed for {subject}: {e}", e)

    return counts


def run_metrics_only(args: Namespace) -> None:
    """Compute ERP/TFR metrics from existing epochs in out_dir/02_epochs."""
    out_dir = Path(args.out_dir)
    epochs_dir = out_dir / "02_epochs"
    metrics_dir = out_dir / "05_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reset_combined_metric_outputs(metrics_dir)

    files = sorted(epochs_dir.glob("*-epo.fif"))
    if not files:
        raise RuntimeError(f"No epochs found in {epochs_dir} (expected *-epo.fif).")

    do_erp = bool(getattr(args, "metrics_erp_enabled", True))
    do_tfr = bool(getattr(args, "metrics_tfr_enabled", True))
    do_ts = bool(getattr(args, "metrics_erp_timeseries", False))
    if not (do_erp or do_tfr or do_ts):
        print("[WARN] Metrics requested but both ERP and TFR are disabled.")
        return

    conditions = resolve_metrics_conditions(args)
    parquet_writer = ParquetRowGroupWriter()
    try:
        for path in files:
            subject = subject_from_epochs_path(path)
            epochs = load_epochs(path).epochs
            counts = write_subject_metrics(
                epochs=epochs,
                subject=subject,
                args=args,
                metrics_dir=metrics_dir,
                conditions=conditions,
                parquet_writer=parquet_writer,
            )
            print(_format_metrics_status(subject, counts))
    finally:
        parquet_writer.close()


def _format_metrics_status(subject: str, counts: dict[str, int]) -> str:
    parts = [f"[OK] {subject}"]
    if "erp_rows" in counts:
        parts.append(f"ERP rows={counts['erp_rows']}")
    if "erp_timeseries_rows" in counts:
        parts.append(f"ERP time-series rows={counts['erp_timeseries_rows']}")
    if "tfr_rows" in counts:
        parts.append(f"TFR rows={counts['tfr_rows']}")
    return " | ".join(parts)


def _handle_metric_failure(args: Namespace, message: str, exc: Exception) -> None:
    print(f"[WARN] {message}")
    if not getattr(args, "allow_metric_failures", False):
        raise RuntimeError(message) from exc
