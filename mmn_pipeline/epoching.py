# mmn_pipeline/epoching.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import mne


@dataclass
class EpochParams:
    tmin: float = -0.2
    tmax: float = 0.6
    baseline: tuple[float, float] = (-0.2, 0.0)


def build_events_from_positions_and_codes(markers_pos: np.ndarray, codes: np.ndarray) -> np.ndarray:
    if len(markers_pos) != len(codes):
        raise ValueError(f"Cannot build events: markers={len(markers_pos)} != codes={len(codes)}")
    events = np.zeros((len(codes), 3), dtype=int)
    events[:, 0] = markers_pos.astype(int)
    events[:, 2] = codes.astype(int)
    return events


def select_and_recode_stddev(
    events: np.ndarray,
    standard_codes: list[int],
    deviant_codes: list[int],
) -> tuple[np.ndarray, dict]:
    std = np.asarray(standard_codes, dtype=int)
    dev = np.asarray(deviant_codes, dtype=int)
    keep = np.isin(events[:, 2], np.r_[std, dev])
    ev2 = events[keep].copy()
    ev2[np.isin(ev2[:, 2], std), 2] = 1
    ev2[np.isin(ev2[:, 2], dev), 2] = 2
    event_id = {"Standard": 1, "Deviant": 2}
    return ev2, event_id


def make_epochs(raw, events_stddev: np.ndarray, event_id: dict, ep: EpochParams):
    epochs = mne.Epochs(
        raw,
        events_stddev,
        event_id=event_id,
        tmin=ep.tmin,
        tmax=ep.tmax,
        baseline=ep.baseline,
        preload=True,
        reject_by_annotation=True,
        detrend=None,
    )
    return epochs