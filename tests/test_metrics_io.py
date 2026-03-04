from pathlib import Path
from types import SimpleNamespace

import pytest

import eeg_pipeline.metrics as metrics_pkg
import eeg_pipeline.metrics.io as metrics_io


def test_metrics_package_exports_expected_symbols():
    assert metrics_pkg.__all__ == ["load_epochs", "compute_erp_metrics", "compute_tfr_metrics"]


def test_load_epochs_requires_existing_file(tmp_path: Path):
    missing = tmp_path / "missing-epo.fif"

    with pytest.raises(FileNotFoundError, match="Epochs file not found"):
        metrics_io.load_epochs(missing)


def test_load_epochs_reads_fif_files(monkeypatch, tmp_path: Path):
    fif_path = tmp_path / "subject-001-epo.fif"
    fif_path.touch()
    sentinel = SimpleNamespace(name="epochs")

    monkeypatch.setattr(metrics_io.mne, "read_epochs", lambda path, preload, verbose: sentinel)

    loaded = metrics_io.load_epochs(fif_path, preload=False)

    assert loaded.epochs is sentinel
    assert loaded.source_path == fif_path
    assert loaded.source_type == "fif"


def test_load_epochs_reads_set_files_and_optionally_preloads(monkeypatch, tmp_path: Path):
    set_path = tmp_path / "subject-001.set"
    set_path.touch()

    class DummyEpochs:
        def __init__(self):
            self.loaded = False

        def load_data(self):
            self.loaded = True

    dummy = DummyEpochs()
    monkeypatch.setattr(metrics_io.mne, "read_epochs_eeglab", lambda path, verbose: dummy)

    loaded = metrics_io.load_epochs(set_path, preload=True)
    assert loaded.epochs is dummy
    assert loaded.source_type == "eeglab"
    assert dummy.loaded is True

    dummy.loaded = False
    loaded = metrics_io.load_epochs(set_path, preload=False)
    assert loaded.epochs is dummy
    assert dummy.loaded is False


def test_load_epochs_rejects_unsupported_extensions(tmp_path: Path):
    txt_path = tmp_path / "epochs.txt"
    txt_path.touch()

    with pytest.raises(ValueError, match="Unsupported epochs format"):
        metrics_io.load_epochs(txt_path)
