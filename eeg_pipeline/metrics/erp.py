# eeg_pipeline/metrics/erp.py
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import mne

from eeg_pipeline.metrics.erp_windows import ERPWindow


def _safe_pick_channels(inst, channels: Sequence[str]) -> list[str]:
    chs = [ch for ch in channels if ch in inst.ch_names]
    if len(chs) == 0:
        raise ValueError(f"None of the requested channels exist in data: {channels}")
    return chs


def _get_evoked(epochs: mne.Epochs, condition: str) -> Optional[mne.Evoked]:
    if condition not in epochs.event_id:
        return None
    ep = epochs[condition]
    if len(ep) == 0:
        return None
    return ep.average()


def _compute_peak(
    data_uv: npt.NDArray[np.floating],
    times: npt.NDArray[np.floating],
    polarity: str,
) -> tuple[float, float]:
    """Compute peak amplitude (µV) and latency (s) within a window."""
    if polarity == "negative":
        idx = int(np.argmin(data_uv))
    elif polarity == "positive":
        idx = int(np.argmax(data_uv))
    else:  # "absolute"
        idx = int(np.argmax(np.abs(data_uv)))

    return float(data_uv[idx]), float(times[idx])


def _rows_for_evoked(
    *,
    evoked: mne.Evoked,
    epochs: mne.Epochs,
    subject: str,
    condition: str,
    channels: Sequence[str],
    windows: Sequence[ERPWindow],
    status: str = "OK",
    source_conditions: str | None = None,
) -> list[dict]:
    # Work on requested channels only (keeps output consistent and predictable)
    pick_chs = _safe_pick_channels(evoked, channels)
    ev = evoked.copy().pick(pick_chs)

    rows: list[dict] = []
    for w in windows:
        ev_crop = ev.copy().crop(tmin=w.tmin, tmax=w.tmax)
        data_uv = ev_crop.data * 1e6  # V → µV
        crop_times = ev_crop.times

        # epoch count: for derived conditions (e.g., MMN), epochs[condition] won't exist
        if condition in epochs.event_id:
            n_epochs = int(len(epochs[condition]))
        else:
            n_epochs = int(len(epochs))

        for ch_idx, ch_name in enumerate(ev_crop.ch_names):
            mean_uv = float(np.mean(data_uv[ch_idx]))
            peak_uv, peak_latency = _compute_peak(
                data_uv[ch_idx],
                crop_times,
                w.polarity,
            )

            rows.append(
                dict(
                    subject=subject,
                    condition=condition,
                    channel=ch_name,
                    component=w.name,
                    window_tmin=float(w.tmin),
                    window_tmax=float(w.tmax),
                    polarity=w.polarity,
                    mean_uV=mean_uv,
                    peak_uV=peak_uv,
                    peak_latency_s=peak_latency,
                    n_epochs=n_epochs,
                    status=status,
                    source_conditions=source_conditions,
                )
            )

    return rows


def compute_erp_metrics(
    epochs: mne.Epochs,
    *,
    subject: str,
    channels: Sequence[str],
    windows: Sequence[ERPWindow],
    conditions: Sequence[str] = ("Standard", "Deviant"),
) -> pd.DataFrame:
    """
    Compute ERP window metrics (mean + peak) per subject × condition × channel × window.

    Output schema is intentionally consistent across all rows:
      subject, condition, channel, component, window_tmin, window_tmax, polarity,
      mean_uV, peak_uV, peak_latency_s, n_epochs, status, source_conditions
    """
    rows: list[dict] = []

    # Condition-specific evokeds
    for cond in conditions:
        ev = _get_evoked(epochs, cond)
        if ev is None:
            continue
        rows.extend(
            _rows_for_evoked(
                evoked=ev,
                epochs=epochs,
                subject=subject,
                condition=cond,
                channels=channels,
                windows=windows,
                status="OK",
                source_conditions=None,
            )
        )

    return pd.DataFrame(rows)
