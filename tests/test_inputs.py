from pathlib import Path

import numpy as np
import pandas as pd

import eeg_pipeline.inputs as inputs


class _FakeRaw:
    def __init__(self):
        self.info = {"sfreq": 100.0}
        self.ch_names = ["Fz", "Cz"]

    def get_channel_types(self):
        return ["eeg", "eeg"]


def _make_legacy_fixture(tmp_path: Path) -> tuple[Path, Path]:
    raw_dir = tmp_path / "legacy_raw"
    subject_csv_dir = tmp_path / "legacy_behavior"
    raw_dir.mkdir(parents=True)
    subject_csv_dir.mkdir(parents=True)

    vhdr = raw_dir / "s01.vhdr"
    vhdr.write_text("MarkerFile=s01.vmrk\nDataFile=s01.eeg\n", encoding="utf-8")
    vhdr.with_suffix(".vmrk").write_text("dummy", encoding="utf-8")
    vhdr.with_suffix(".eeg").write_text("dummy", encoding="utf-8")
    (subject_csv_dir / "subject-01.csv").write_text("EventCode\n1\n2\n", encoding="utf-8")
    return raw_dir, subject_csv_dir


def test_discover_pipeline_recordings_legacy_uses_csv_sidecars(tmp_path: Path):
    raw_dir, subject_csv_dir = _make_legacy_fixture(tmp_path)

    recordings = inputs.discover_pipeline_recordings(
        mode="legacy",
        bids_root=None,
        raw_dir=raw_dir,
        subject_csv_dir=subject_csv_dir,
        task_label="oddball",
    )

    assert len(recordings) == 1
    recording = recordings[0]
    assert recording.source_type == "legacy"
    assert recording.entities == {"sub": "01", "task": "oddball"}
    assert recording.behavior_path == subject_csv_dir / "subject-01.csv"
    assert recording.behavior_kind == "csv"


def test_convert_legacy_recordings_to_bids_writes_raw_bids_fixture(monkeypatch, tmp_path: Path):
    raw_dir, subject_csv_dir = _make_legacy_fixture(tmp_path)
    recordings = inputs.discover_legacy_recordings(
        raw_dir,
        subject_csv_dir=subject_csv_dir,
        task_label="oddball",
    )
    bids_root = tmp_path / "converted_bids"

    monkeypatch.setattr(inputs, "_read_raw_minimal", lambda raw_path: _FakeRaw())
    monkeypatch.setattr(
        inputs,
        "events_from_annotations_positions",
        lambda raw: np.array([[0, 0, 1], [100, 0, 2]], dtype=int),
    )

    converted = inputs.convert_legacy_recordings_to_bids(
        recordings,
        bids_root=bids_root,
        task_label="oddball",
        keep_codes=[1, 2],
        standard_codes=[1],
        deviant_codes=[2],
        drop_eeg_markers_by_gap_s=None,
        auto_drop_to_count=True,
    )

    eeg_dir = bids_root / "sub-01" / "eeg"
    events_tsv = eeg_dir / "sub-01_task-oddball_events.tsv"

    assert (bids_root / "dataset_description.json").exists()
    assert (bids_root / "participants.tsv").exists()
    assert (eeg_dir / "sub-01_task-oddball_eeg.vhdr").exists()
    assert events_tsv.exists()
    assert (eeg_dir / "sub-01_task-oddball_events.json").exists()

    events_df = pd.read_csv(events_tsv, sep="\t")
    assert events_df["value"].tolist() == [1, 2]
    assert events_df["trial_type"].tolist() == ["Standard", "Deviant"]
    assert converted[0].source_type == "bids"
    assert converted[0].behavior_kind == "bids_events"
