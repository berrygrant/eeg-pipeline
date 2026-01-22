# eeg_pipeline/metrics/io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mne


@dataclass
class LoadedEpochs:
    epochs: mne.Epochs
    source_path: Path
    source_type: str  # "fif" or "eeglab"


def load_epochs(path: str | Path, preload: bool = True) -> LoadedEpochs:
    """
    Load epochs from either:
      - MNE FIF epochs: *.fif (expects *-epo.fif)
      - EEGLAB epochs: *.set

    Returns a LoadedEpochs wrapper including source info.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Epochs file not found: {p}")

    suf = p.suffix.lower()
    if suf == ".fif":
        epochs = mne.read_epochs(p, preload=preload, verbose="error")
        return LoadedEpochs(epochs=epochs, source_path=p, source_type="fif")

    if suf == ".set":
        epochs = mne.read_epochs_eeglab(p, verbose="error")
        if preload:
            epochs.load_data()
        return LoadedEpochs(epochs=epochs, source_path=p, source_type="eeglab")

    raise ValueError(f"Unsupported epochs format: {p.suffix} (expected .fif or .set)")