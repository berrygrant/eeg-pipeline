# eeg_pipeline/metrics/erp_timeseries.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import mne


@dataclass
class ERPTimeSeriesParams:
    tmin: float = -0.2
    tmax: float = 0.6
    baseline: tuple[float, float] | None = (-0.2, 0.0)
    decim: int = 1  # optional downsampling of time points for smaller files


def _safe_pick_channels(inst: mne.Epochs | mne.Evoked, channels: Sequence[str]) -> list[str]:
    keep = [ch for ch in channels if ch in inst.ch_names]
    if len(keep) == 0:
        raise ValueError(f"None of requested channels found. Requested={channels}.")
    return keep


def _evoked_to_long_df(
    ev: mne.Evoked,
    *,
    subject: str,
    condition: str,
    n_epochs: int,
    decim: int = 1,
) -> pd.DataFrame:
    """Convert an Evoked to tidy long time-series DataFrame."""
    ev2 = ev.copy()
    if decim and decim > 1:
        ev2 = ev2.decimate(decim, offset=0)

    times = ev2.times  # seconds, shape (n_times,)
    data_uv = ev2.data * 1e6  # (n_ch, n_times)

    n_ch, n_times = data_uv.shape
    # Build long form without python loops for speed
    df = pd.DataFrame(
        {
            "subject": subject,
            "condition": condition,
            "n_epochs": n_epochs,
            "channel": np.repeat(ev2.ch_names, n_times),
            "time_s": np.tile(times, n_ch),
            "amplitude_uv": data_uv.reshape(-1, order="C"),
        }
    )
    return df


def compute_erp_timeseries(
    epochs: mne.Epochs,
    *,
    subject: str,
    channels: Sequence[str],
    params: ERPTimeSeriesParams = ERPTimeSeriesParams(),
    conditions: Sequence[str] = ("Standard", "Deviant"),
    include_difference_wave: bool = True,
    difference_label: str = "MMN_DEV_MINUS_STD",
) -> pd.DataFrame:
    """
    Compute time-series ERPs per subject × condition × channel × time.

    Returns tidy long DataFrame.
    """
    if epochs is None or len(epochs) == 0:
        return pd.DataFrame([{"subject": subject, "status": "EMPTY_EPOCHS"}])

    chs = _safe_pick_channels(epochs, channels)

    ep = (
        epochs.copy()
        .pick(chs)
        .crop(tmin=params.tmin, tmax=params.tmax)
    )

    if params.baseline is not None:
        ep.apply_baseline(params.baseline)

    out = []
    ev_by_cond: dict[str, mne.Evoked] = {}
    n_by_cond: dict[str, int] = {}

    for cond in conditions:
        if cond not in ep.event_id:
            continue
        ep_c = ep[cond]
        n = len(ep_c)
        if n == 0:
            continue
        ev = ep_c.average()
        ev_by_cond[cond] = ev
        n_by_cond[cond] = n
        out.append(
            _evoked_to_long_df(
                ev,
                subject=subject,
                condition=cond,
                n_epochs=n,
                decim=params.decim,
            )
        )

    # Optional difference wave (Deviant - Standard)
    if include_difference_wave and ("Deviant" in ev_by_cond) and ("Standard" in ev_by_cond):
        ev_dev = ev_by_cond["Deviant"]
        ev_std = ev_by_cond["Standard"]
        ev_diff = ev_dev.copy()
        ev_diff.data = ev_dev.data - ev_std.data
        out.append(
            _evoked_to_long_df(
                ev_diff,
                subject=subject,
                condition=difference_label,
                n_epochs=min(n_by_cond["Deviant"], n_by_cond["Standard"]),
                decim=params.decim,
            )
        )

    if not out:
        return pd.DataFrame([{"subject": subject, "status": "NO_CONDITIONS"}])

    df = pd.concat(out, ignore_index=True)
    df["status"] = "OK"
    return df