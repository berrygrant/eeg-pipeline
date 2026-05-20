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


def _empty_status_row(subject: str, condition: str, status: str) -> dict:
    return dict(
        subject=subject,
        condition=condition,
        channel="",
        frequency=np.nan,
        time=np.nan,
        total_power=np.nan,
        evoked_power=np.nan,
        induced_power=np.nan,
        itc=np.nan,
        n_epochs=0,
        status=status,
    )


def _tfr_condition_frame(
    *,
    subject: str,
    condition: str,
    ch_names: Sequence[str],
    freqs: np.ndarray,
    times: np.ndarray,
    total_power: np.ndarray,
    evoked_power: np.ndarray,
    itc: np.ndarray,
    n_epochs: int,
) -> pd.DataFrame:
    n_ch = len(ch_names)
    n_freq = len(freqs)
    n_times = len(times)

    total_flat = total_power.reshape(-1)
    evoked_flat = evoked_power.reshape(-1)

    return pd.DataFrame(
        {
            "subject": subject,
            "condition": condition,
            "channel": np.repeat(np.asarray(ch_names, dtype=object), n_freq * n_times),
            "frequency": np.tile(np.repeat(freqs.astype(float), n_times), n_ch),
            "time": np.tile(times.astype(float), n_ch * n_freq),
            "total_power": total_flat,
            "evoked_power": evoked_flat,
            "induced_power": total_flat - evoked_flat,
            "itc": itc.reshape(-1),
            "n_epochs": int(n_epochs),
            "status": "OK",
        }
    )


def compute_tfr_metrics(
    epochs: mne.Epochs,
    *,
    subject: str,
    channels: Sequence[str],
    tmin: float,
    tmax: float,
    conditions: Sequence[str] = ("Standard", "Deviant"),
    params: TFRParams = TFRParams(),
    time_decim: int = 1,
) -> pd.DataFrame:
    """
    Computes per condition:
      - total_power: TFR of epochs (averaged over trials)
      - evoked_power: TFR of evoked (TFR of the average)
      - induced_power = total_power - evoked_power
      - itc: inter-trial coherence from epochs TFR

    Output is a tidy DataFrame (subject × condition × channel × freq × time).
    """

    if epochs is None or len(epochs) == 0:
        return pd.DataFrame([_empty_status_row(subject, "", "EMPTY_EPOCHS_OBJECT")])

    channels = _safe_pick_channels(epochs, channels)

    # Work on cropped/picked copy (requires non-empty + preload)
    ep = epochs.copy().crop(tmin=tmin, tmax=tmax).pick(channels)

    freqs = np.arange(params.fmin, params.fmax + params.fstep, params.fstep, dtype=float)

    frames: list[pd.DataFrame] = []

    for cond in conditions:
        if cond not in ep.event_id:
            frames.append(pd.DataFrame([_empty_status_row(subject, cond, "MISSING_CONDITION")]))
            continue

        ep_cond = ep[cond]
        if len(ep_cond) == 0:
            frames.append(pd.DataFrame([_empty_status_row(subject, cond, "EMPTY")]))
            continue

        # Total power + ITC (averaged across epochs)
        power_total, itc = _compute_tfr_epochs(ep_cond, freqs, params)

        # ERPLAB-style: no TF-domain baseline for power or ITC

        # Evoked power (power of the evoked response)
        ev = ep_cond.average()
        tfr_evoked = _compute_tfr_evoked(ev, freqs, params)

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

        t_step = max(1, int(time_decim))
        time_idx = np.arange(0, len(power_total.times), t_step, dtype=int)
        frames.append(
            _tfr_condition_frame(
                subject=subject,
                condition=cond,
                ch_names=power_total.ch_names,
                freqs=power_total.freqs,
                times=power_total.times[time_idx],
                total_power=power_total.data[:, :, time_idx],
                evoked_power=tfr_evoked.data[:, :, time_idx],
                itc=itc.data[:, :, time_idx],
                n_epochs=len(ep_cond),
            )
        )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
