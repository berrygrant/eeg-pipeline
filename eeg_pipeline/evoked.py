# mmn_pipeline/evoked.py
from __future__ import annotations

import mne


def compute_evokeds(epochs):
    evo_std = epochs["Standard"].average()
    evo_dev = epochs["Deviant"].average()
    return evo_std, evo_dev


def grand_averages(evokeds_std, evokeds_dev):
    ga_std = mne.grand_average(evokeds_std)
    ga_dev = mne.grand_average(evokeds_dev)
    return ga_std, ga_dev