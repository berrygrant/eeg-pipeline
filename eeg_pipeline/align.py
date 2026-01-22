# mmn_pipeline/align.py
from __future__ import annotations

import numpy as np


def marker_gap_stats(markers_pos: np.ndarray, sfreq: float) -> dict:
    if markers_pos.size < 2:
        return {"n": int(markers_pos.size)}
    t = markers_pos / sfreq
    dt = np.diff(t)
    q = np.quantile(dt, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return {
        "n": int(markers_pos.size),
        "dt_min": float(q[0]),
        "dt_p25": float(q[1]),
        "dt_p50": float(q[2]),
        "dt_p75": float(q[3]),
        "dt_p90": float(q[4]),
        "dt_p95": float(q[5]),
        "dt_p99": float(q[6]),
        "dt_max": float(q[7]),
    }


def keep_by_gap_heuristic(markers_pos: np.ndarray, sfreq: float, gap_s: float) -> np.ndarray:
    if markers_pos.size == 0:
        return np.array([], dtype=int)
    t = markers_pos / sfreq
    dt_prev = np.r_[np.inf, np.diff(t)]
    dt_next = np.r_[np.diff(t), np.inf]
    keep = (dt_prev <= gap_s) & (dt_next <= gap_s)
    return np.where(keep)[0]


def keep_best_dense_markers_to_count(markers_pos: np.ndarray, sfreq: float, target_n: int) -> np.ndarray:
    """
    Keep exactly target_n markers that are most embedded in the dense stream.

    score_i = max(dt_prev_i, dt_next_i) in seconds; keep lowest scores.
    """
    n = markers_pos.size
    if target_n > n:
        raise ValueError(f"target_n ({target_n}) > n_markers ({n})")
    if target_n == n:
        return np.arange(n, dtype=int)

    t = markers_pos / sfreq
    dt_prev = np.r_[np.inf, np.diff(t)]
    dt_next = np.r_[np.diff(t), np.inf]
    score = np.maximum(dt_prev, dt_next)

    idx_sorted = np.argsort(score, kind="mergesort")
    keep_unsorted = np.sort(idx_sorted[:target_n])
    return keep_unsorted


def align_marker_positions_to_codes(
    markers_pos: np.ndarray,
    sfreq: float,
    codes: np.ndarray,
    gap_s: float | None,
    auto_drop_to_count: bool,
) -> tuple[np.ndarray, dict]:
    """
    Returns:
      markers_aligned (len == len(codes)),
      diag dict
    """
    diag = {
        "markers_original": int(len(markers_pos)),
        "markers_dropped_by_gap": 0,
        "markers_dropped_by_auto": 0,
    }

    idx = np.arange(len(markers_pos), dtype=int)

    if gap_s is not None:
        keep_idx = keep_by_gap_heuristic(markers_pos, sfreq=sfreq, gap_s=float(gap_s))
        diag["markers_dropped_by_gap"] = int(len(markers_pos) - len(keep_idx))
        idx = keep_idx

    markers_pos2 = markers_pos[idx]

    if len(markers_pos2) == len(codes):
        return markers_pos2, diag

    if len(markers_pos2) > len(codes) and auto_drop_to_count:
        keep2 = keep_best_dense_markers_to_count(markers_pos2, sfreq=sfreq, target_n=len(codes))
        diag["markers_dropped_by_auto"] = int(len(markers_pos2) - len(keep2))
        markers_pos2 = markers_pos2[keep2]
        return markers_pos2, diag

    raise ValueError(
        f"Alignment failed: EEG markers={len(markers_pos2)} vs behavioral codes={len(codes)}. "
        f"Try --behavioral_keep_codes and/or --drop_eeg_markers_by_gap_s (or enable auto-drop)."
    )