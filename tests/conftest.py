import sys
from pathlib import Path

import mne
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def synthetic_raw():
    sfreq = 100.0
    times = np.arange(1000, dtype=float) / sfreq
    data = np.vstack(
        [
            1e-6 * np.sin(2 * np.pi * 5 * times),
            1e-6 * np.cos(2 * np.pi * 3 * times),
            2e-6 * np.sin(2 * np.pi * 1 * times),
        ]
    )
    info = mne.create_info(["Fz", "Cz", "EOG"], sfreq, ch_types=["eeg", "eeg", "eog"])
    return mne.io.RawArray(data, info, verbose="error")


@pytest.fixture
def synthetic_epochs(synthetic_raw):
    events = np.array(
        [
            [100, 0, 1],
            [300, 0, 2],
            [500, 0, 1],
            [700, 0, 2],
        ],
        dtype=int,
    )
    return mne.Epochs(
        synthetic_raw.copy(),
        events,
        event_id={"Standard": 1, "Deviant": 2},
        tmin=-0.1,
        tmax=0.2,
        baseline=None,
        preload=True,
        reject_by_annotation=False,
        on_missing="warn",
        detrend=None,
        verbose="error",
    )
