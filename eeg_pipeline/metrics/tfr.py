# eeg_pipeline/metrics/tfr.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional

import numpy as np
import pandas as pd
import mne


@dataclass(frozen=True)
class TFRParams:
    fmin: float = 1.0
    fmax: float = 30.0
    fstep: float = 1.0
    method: str = "multitaper"  # "morlet" or "multitaper"
    n_cycles_div: float = 10.0  # n_cycles = freqs / n_cycles_div
    decim: int = 1
    baseline: Optional[tuple[float, float]] = (-0.1, 0.0)
    mode: str = "logratio"  # apply_baseline mode


def _safe_pick_channels(inst, channels: Sequence[str]) -> list[str]:
    chs = [ch for ch in channels if ch in inst.ch_names]
    if len(chs) == 0:
        raise ValueError(f"None of the requested channels exist in data: {channels}")
    return chs


def _compute_tfr_epochs(epochs: mne.Epochs, freqs: np.ndarray, params: TFRParams):
    n_cycles = freqs / float(params.n_cycles_div)

    # Epochs.compute_tfr supports return_itc and average
    power, itc = epochs.compute_tfr(
        method=params.method,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=True,
        decim=params.decim,
        average=True,
    )
    return power, itc


def _compute_tfr_evoked(evoked: mne.Evoked, freqs: np.ndarray, params: TFRParams):
    """
    IMPORTANT:
    Evoked.compute_tfr() has a different signature than Epochs.compute_tfr().
    In particular, it does NOT accept return_itc or average in many MNE versions.
    """
    n_cycles = freqs / float(params.n_cycles_div)

    # Evoked.compute_tfr returns an AverageTFR
    tfr = evoked.compute_tfr(
        method=params.method,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        decim=params.decim,
    )
    return tfr


def compute_tfr_metrics(
    epochs: mne.Epochs,
    *,
    subject: str,
    channels: Sequence[str],
    tmin: float,
    tmax: float,
    conditions: Sequence[str] = ("Standard", "Deviant"),
    params: TFRParams = TFRParams(),
) -> pd.DataFrame:
    """
    Computes per condition:
      - total_power: TFR of epochs (averaged over trials)
      - evoked_power: TFR of evoked (TFR of the average)
      - induced_power = total_power - evoked_power
      - itc: inter-trial coherence from epochs TFR

    Output is a tidy DataFrame (subject × condition × channel × freq × time).
    """

    # ---- NEW: bail out early on empty epochs objects ----
    if epochs is None or len(epochs) == 0:
        return pd.DataFrame(
            [
                dict(
                    subject=subject,
                    condition="",
                    channel="",
                    frequency=np.nan,
                    time=np.nan,
                    total_power=np.nan,
                    evoked_power=np.nan,
                    induced_power=np.nan,
                    itc=np.nan,
                    n_epochs=0,
                    status="EMPTY_EPOCHS_OBJECT",
                )
            ]
        )

    channels = _safe_pick_channels(epochs, channels)

    # Work on cropped/picked copy (requires non-empty + preload)
    ep = epochs.copy().crop(tmin=tmin, tmax=tmax).pick(channels)

    # Baseline on epochs before TF (common workflow)
    if params.baseline is not None:
        ep.apply_baseline(params.baseline)

    freqs = np.arange(params.fmin, params.fmax + params.fstep, params.fstep, dtype=float)

    rows: list[dict] = []

    for cond in conditions:
        if cond not in ep.event_id:
            rows.append(
                dict(
                    subject=subject,
                    condition=cond,
                    channel="",
                    frequency=np.nan,
                    time=np.nan,
                    total_power=np.nan,
                    evoked_power=np.nan,
                    induced_power=np.nan,
                    itc=np.nan,
                    n_epochs=0,
                    status="MISSING_CONDITION",
                )
            )
            continue

        ep_cond = ep[cond]
        if len(ep_cond) == 0:
            rows.append(
                dict(
                    subject=subject,
                    condition=cond,
                    channel="",
                    frequency=np.nan,
                    time=np.nan,
                    total_power=np.nan,
                    evoked_power=np.nan,
                    induced_power=np.nan,
                    itc=np.nan,
                    n_epochs=0,
                    status="EMPTY",
                )
            )
            continue

        # Total power + ITC (averaged across epochs)
        power_total, itc = _compute_tfr_epochs(ep_cond, freqs, params)

        # Baseline in TF domain (logratio etc.)
        if params.baseline is not None:
            power_total.apply_baseline(params.baseline, mode=params.mode)
            itc.apply_baseline(params.baseline, mode=params.mode)

        # Evoked power (power of the evoked response)
        ev = ep_cond.average()
        tfr_evoked = _compute_tfr_evoked(ev, freqs, params)
        if params.baseline is not None:
            tfr_evoked.apply_baseline(params.baseline, mode=params.mode)

        # Safety: ensure identical axes before subtraction
        if (
            power_total.data.shape != tfr_evoked.data.shape
            or not np.allclose(power_total.times, tfr_evoked.times)
            or not np.allclose(power_total.freqs, tfr_evoked.freqs)
            or power_total.ch_names != tfr_evoked.ch_names
        ):
            raise RuntimeError(
                "Total-power TFR and evoked-power TFR axes do not match; cannot compute induced power safely."
            )

        induced = power_total.copy()
        induced.data = power_total.data - tfr_evoked.data

        # Flatten to rows
        for ch_i, ch in enumerate(power_total.ch_names):
            for f_i, f in enumerate(power_total.freqs):
                for t_i, tt in enumerate(power_total.times):
                    rows.append(
                        dict(
                            subject=subject,
                            condition=cond,
                            channel=ch,
                            frequency=float(f),
                            time=float(tt),
                            total_power=float(power_total.data[ch_i, f_i, t_i]),
                            evoked_power=float(tfr_evoked.data[ch_i, f_i, t_i]),
                            induced_power=float(induced.data[ch_i, f_i, t_i]),
                            itc=float(itc.data[ch_i, f_i, t_i]),
                            n_epochs=int(len(ep_cond)),
                            status="OK",
                        )
                    )

    return pd.DataFrame(rows)