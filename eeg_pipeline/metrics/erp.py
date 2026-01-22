# eeg_pipeline/metrics/erp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional

import numpy as np
import pandas as pd
import mne


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
    """
    Compute mean amplitude per channel per window for specified conditions.

    If compute_mmn=True and both "Standard" and "Deviant" exist, also computes
    a difference wave: Deviant - Standard using mne.combine_evoked.

    Returns a tidy DataFrame with one row per:
      subject × condition × channel × window
    """
    channels = _safe_pick_channels(epochs, channels)

    # Build evokeds
    evokeds = {}
    for cond in conditions:
        ev = _get_evoked(epochs, cond)
        evokeds[cond] = ev

    if compute_mmn:
        ev_std = evokeds.get("Standard")
        ev_dev = evokeds.get("Deviant")
        if ev_std is not None and ev_dev is not None:
            evokeds[mmn_name] = mne.combine_evoked([ev_dev, ev_std], weights=[1.0, -1.0])
        else:
            evokeds[mmn_name] = None

    rows: list[dict] = []

    for cond, ev in evokeds.items():
        if ev is None:
            # still emit a status row so your batch runs don’t “silently drop”
            rows.append(
                dict(
                    subject=subject,
                    condition=cond,
                    channel="",
                    window="",
                    tmin=np.nan,
                    tmax=np.nan,
                    mean_uV=np.nan,
                    n_epochs=int(len(epochs[cond])) if cond in epochs.event_id else 0,
                    status="MISSING_OR_EMPTY",
                )
            )
            continue

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