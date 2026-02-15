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


def select_and_filter_conditions(
    events: np.ndarray,
    condition_map: dict,
) -> tuple[np.ndarray, dict, list[int]]:
    """
    Filter events to those in condition_map and build event_id.

    condition_map should map condition name -> int code (single code per condition).
    Returns (events_filtered, event_id, codes_flat).
    """
    if not isinstance(condition_map, dict) or not condition_map:
        raise ValueError("condition_map must be a non-empty dict of name -> code.")

    event_id: dict[str, int] = {}
    codes_flat: list[int] = []
    seen: set[int] = set()

    for name, code in condition_map.items():
        if isinstance(code, (list, tuple, set)):
            codes = [int(c) for c in code]
        else:
            codes = [int(code)]
        if len(codes) != 1:
            raise ValueError(
                f"condition_map['{name}'] must map to a single code; got {codes}."
            )
        c = int(codes[0])
        if c in seen:
            raise ValueError(f"Duplicate code in condition_map: {c}")
        seen.add(c)
        event_id[str(name)] = c
        codes_flat.append(c)

    keep = np.isin(events[:, 2], np.asarray(codes_flat, dtype=int))
    ev2 = events[keep].copy()
    return ev2, event_id, codes_flat


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
        on_missing="warn",
        detrend=None,
    )
    return epochs
