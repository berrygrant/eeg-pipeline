# eeg_pipeline/metrics/erp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional

import numpy as np
import pandas as pd
import mne
import numpy.typing as npt
from eeg_pipeline.metrics.erp_windows import ERPWindow

@dataclass(frozen=True)
class ERPWindow:
    """
    An ERP measurement window.
    name: label for the component (e.g., "MMN_150_250")
    tmin, tmax: seconds
    """
    name: str
    tmin: float
    tmax: float


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
    """
    Compute peak amplitude and latency according to polarity.
    """
    if polarity == "negative":
        idx = int(np.argmin(data_uv))
    elif polarity == "positive":
        idx = int(np.argmax(data_uv))
    else:  # absolute
        idx = int(np.argmax(np.abs(data_uv)))

    return float(data_uv[idx]), float(times[idx])

def compute_erp_metrics(
    epochs: mne.Epochs,
    *,
    subject: str,
    channels: Sequence[str],
    windows: Sequence[ERPWindow],
    conditions: Sequence[str] = ("Standard", "Deviant"),
    compute_mmn: bool = True,
    mmn_name: str = "MMN",
    ) -> pd.DataFrame:
    times = epochs.times
    rows=[]
    for cond in conditions:
        ev = _get_evoked(epochs, cond)
        if ev is None:
            continue

        for w in windows:
            ev_crop = ev.copy().crop(tmin=w.tmin, tmax=w.tmax)
            data_uv = ev_crop.data * 1e6  # V → µV
            crop_times = ev_crop.times

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
                        condition=cond,
                        channel=ch_name,
                        component=w.name,
                        window_tmin=w.tmin,
                        window_tmax=w.tmax,
                        polarity=w.polarity,
                        mean_uV=mean_uv,
                        peak_uV=peak_uv,
                        peak_latency_s=peak_latency,
                        n_epochs=int(len(epochs[cond]))
                        if cond in epochs.event_id
                        else int(len(epochs)),
                        status="OK",
                    )
                )

        ev_pick = ev.copy().pick(channels)

        for w in windows:
            crop = ev_pick.copy().crop(tmin=w.tmin, tmax=w.tmax)
            # crop.data is (n_ch, n_times) in Volts
            mean_v = crop.data.mean(axis=1)  # per channel
            mean_uv = mean_v * 1e6

            for ch, val in zip(crop.ch_names, mean_uv):
                rows.append(
                    dict(
                        subject=subject,
                        condition=cond,
                        channel=ch,
                        window=w.name,
                        tmin=w.tmin,
                        tmax=w.tmax,
                        mean_uV=float(val),
                        n_epochs=int(len(epochs[cond])) if cond in epochs.event_id else int(len(epochs)),
                        status="OK",
                    )
                )

    return pd.DataFrame(rows)