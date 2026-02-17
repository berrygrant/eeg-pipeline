# mmn_pipeline/evoked.py
from __future__ import annotations

import mne


def compute_evokeds(epochs, conditions):
    """
    Compute evokeds for each condition present in epochs.event_id.
    Returns dict: condition -> Evoked.
    """
    evokeds = {}
    for cond in conditions:
        if cond not in epochs.event_id:
            continue
        if len(epochs[cond]) == 0:
            continue
        evokeds[str(cond)] = epochs[cond].average()
    return evokeds


def grand_averages(evokeds_by_cond, *, weighting: str = "equal"):
    """
    Compute grand averages for each condition.
    evokeds_by_cond: dict[str, list[Evoked]]
    Returns dict[str, Evoked].
    """
    ga = {}
    weighting = str(weighting).lower()
    for cond, ev_list in evokeds_by_cond.items():
        if not ev_list:
            continue
        if weighting == "nave":
            ga[cond] = mne.combine_evoked(ev_list, weights="nave")
        else:
            ga[cond] = mne.grand_average(ev_list)
    return ga
