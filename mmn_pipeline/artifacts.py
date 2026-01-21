# mmn_pipeline/artifacts.py
from __future__ import annotations

import numpy as np


def uv_to_v(x_uv: float) -> float:
    return x_uv * 1e-6


def moving_window_ptp_mask(
    data_v: np.ndarray,
    *,
    sfreq: float,
    win_ms: float,
    step_ms: float,
    threshold_uv: float,
) -> np.ndarray:
    """
    Moving-window peak-to-peak detector.

    Accepts either:
      - epochs-shaped data: (n_epochs, n_ch, n_times) -> returns (n_epochs,) bool
      - raw/continuous data: (n_ch, n_times)         -> returns (n_windows,) bool

    For epochs, each epoch is flagged if ANY channel exceeds threshold in ANY window.
    For raw, each window is flagged if ANY channel exceeds threshold in that window.
    """

    data = np.asarray(data_v)

    win_samp = max(1, int(round((win_ms / 1000.0) * sfreq)))
    step_samp = max(1, int(round((step_ms / 1000.0) * sfreq)))
    thr_v = float(threshold_uv) * 1e-6

    if data.ndim == 2:
        # (n_ch, n_times)
        n_ch, n_times = data.shape
        if n_times < win_samp:
            return np.zeros(0, dtype=bool)

        starts = np.arange(0, n_times - win_samp + 1, step_samp, dtype=int)
        bad = np.zeros(len(starts), dtype=bool)

        for i, st in enumerate(starts):
            seg = data[:, st : st + win_samp]  # (n_ch, win_samp)
            ptp = seg.max(axis=1) - seg.min(axis=1)  # (n_ch,)
            bad[i] = bool(np.any(ptp > thr_v))
        return bad

    if data.ndim == 3:
        # (n_epochs, n_ch, n_times)
        n_epochs, n_ch, n_times = data.shape
        if n_times < win_samp:
            return np.zeros(n_epochs, dtype=bool)

        starts = np.arange(0, n_times - win_samp + 1, step_samp, dtype=int)
        bad = np.zeros(n_epochs, dtype=bool)

        for st in starts:
            seg = data[:, :, st : st + win_samp]  # (n_epochs, n_ch, win_samp)
            ptp = seg.max(axis=2) - seg.min(axis=2)  # (n_epochs, n_ch)
            bad |= np.any(ptp > thr_v, axis=1)
        return bad

    raise ValueError(f"moving_window_ptp_mask expected 2D or 3D array, got shape {data.shape}")


def simple_voltage_threshold_mask(
    data_v: np.ndarray,
    pos_limit_uv: float,
    neg_limit_uv: float,
) -> np.ndarray:
    """
    Simple max/min thresholding across the entire epoch.
    data_v: (n_epochs, n_channels, n_times) in Volts.
    """
    pos_v = uv_to_v(pos_limit_uv)
    neg_v = uv_to_v(neg_limit_uv)
    mx = data_v.max(axis=2)
    mn = data_v.min(axis=2)
    return (mx > pos_v).any(axis=1) | (mn < neg_v).any(axis=1)