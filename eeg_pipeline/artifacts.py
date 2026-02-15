# mmn_pipeline/artifacts.py
from __future__ import annotations

import numpy as np

from .gpu import get_xp, to_numpy, backend as gpu_backend


def _to_uV(x: np.ndarray) -> np.ndarray:
    """MNE returns EEG/EOG in Volts; convert to microvolts."""
    return x * 1e6


def _maybe_to_numpy(x):
    if gpu_backend() == "cupy":
        return to_numpy(x)
    return x


def moving_window_ptp_mask(
    data_v: np.ndarray,
    sfreq: float,
    win_ms: float,
    step_ms: float,
    threshold_uv: float,
) -> np.ndarray:
    """
    Returns boolean mask (n_epochs,) where True indicates epoch is bad.

    data_v can be:
      - (n_epochs, n_ch, n_times)
      - (n_epochs, n_times)  [single-channel already selected]
    """
    xp = get_xp()
    x = _to_uV(xp.asarray(data_v))

    if x.ndim == 2:
        x = x[:, None, :]  # (n_epochs, 1, n_times)
    if x.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape={x.shape}")

    n_epochs, n_ch, n_times = x.shape

    win_samp = max(1, int(round((win_ms / 1000.0) * sfreq)))
    step_samp = max(1, int(round((step_ms / 1000.0) * sfreq)))
    if win_samp > n_times:
        # Degenerate: window larger than data
        ptp = x.max(axis=-1) - x.min(axis=-1)  # (n_epochs, n_ch)
        return _maybe_to_numpy((ptp.max(axis=1) >= threshold_uv))

    bad = xp.zeros(n_epochs, dtype=bool)
    for start in range(0, n_times - win_samp + 1, step_samp):
        seg = x[:, :, start : start + win_samp]  # (n_epochs, n_ch, win)
        ptp = seg.max(axis=-1) - seg.min(axis=-1)  # (n_epochs, n_ch)
        bad |= (ptp.max(axis=1) >= threshold_uv)
    return _maybe_to_numpy(bad)


def simple_voltage_threshold_mask(
    data_v: np.ndarray,
    pos_limit_uv: float,
    neg_limit_uv: float,
) -> np.ndarray:
    """
    Returns boolean mask (n_epochs,) where True indicates epoch is bad.

    data_v can be:
      - (n_epochs, n_ch, n_times)
      - (n_epochs, n_times)
    """
    xp = get_xp()
    x = _to_uV(xp.asarray(data_v))

    if x.ndim == 2:
        x = x[:, None, :]
    if x.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape={x.shape}")

    maxv = x.max(axis=-1).max(axis=-1)  # (n_epochs,)
    minv = x.min(axis=-1).min(axis=-1)  # (n_epochs,)
    bad = (maxv >= pos_limit_uv) | (minv <= neg_limit_uv)
    return _maybe_to_numpy(bad)
