from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

from eeg_pipeline.manual_review import (
    apply_sidecar_to_epochs,
    apply_sidecar_to_raw,
    default_sidecar_path,
    load_sidecar,
    save_sidecar,
)


def _synthetic_raw() -> mne.io.BaseRaw:
    sfreq = 100.0
    data = np.zeros((2, int(sfreq * 2.5)))
    info = mne.create_info(["Cz", "Pz"], sfreq=sfreq, ch_types=["eeg", "eeg"])
    raw = mne.io.RawArray(data, info, verbose="error")
    return raw


def _synthetic_epochs() -> mne.Epochs:
    raw = _synthetic_raw()
    events = np.array(
        [
            [50, 0, 1],
            [100, 0, 1],
            [150, 0, 1],
            [200, 0, 1],
        ],
        dtype=int,
    )
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={"Stim": 1},
        tmin=-0.1,
        tmax=0.1,
        baseline=None,
        preload=True,
        reject_by_annotation=False,
        verbose="error",
    )
    return epochs


def test_default_sidecar_path_and_roundtrip(tmp_path: Path):
    input_path = tmp_path / "sub-001-epo.fif"
    payload = {
        "schema_version": 1,
        "mode": "epochs",
        "bad_channels": ["Cz"],
        "dropped_epoch_indices": [1, 3],
    }
    sidecar = default_sidecar_path(input_path)
    assert sidecar.name == "sub-001-epo.fif.manual_reject.json"

    save_sidecar(payload, sidecar)
    loaded = load_sidecar(sidecar)
    assert loaded == payload


def test_apply_sidecar_to_raw_adds_bads_and_deduplicates_annotations():
    raw = _synthetic_raw()
    raw.info["bads"] = ["Cz"]
    raw.set_annotations(mne.Annotations(onset=[0.5], duration=[0.1], description=["BAD_manual"]))

    payload = {
        "bad_channels": ["Pz", "Missing"],
        "annotations": [
            {"onset_s": 0.5, "duration_s": 0.1, "description": "BAD_manual"},
            {"onset_s": 1.0, "duration_s": 0.2, "description": "BAD_manual"},
        ],
    }
    stats = apply_sidecar_to_raw(raw, payload)

    assert stats["bad_channels_added"] == 1
    assert stats["annotations_added"] == 1
    assert sorted(raw.info["bads"]) == ["Cz", "Pz"]
    assert len(raw.annotations) == 2


def test_apply_sidecar_to_epochs_uses_selection_indices_and_adds_bad_channels():
    epochs = _synthetic_epochs()
    # Simulate a pre-existing drop so selection is no longer 0..N-1.
    epochs.drop([0], reason="PREEXIST")
    assert epochs.selection.tolist() == [1, 2, 3]

    payload = {
        "bad_channels": ["Pz", "Unknown"],
        "dropped_epoch_indices": [2, 999],
    }
    stats = apply_sidecar_to_epochs(epochs, payload)

    assert stats["bad_channels_added"] == 1
    assert stats["epochs_dropped"] == 1
    assert stats["epochs_requested"] == 2
    assert epochs.selection.tolist() == [1, 3]
    assert "Pz" in epochs.info["bads"]
