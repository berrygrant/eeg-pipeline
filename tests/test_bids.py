import json
from pathlib import Path

from eeg_pipeline.bids import (
    BIDSRecording,
    discover_bids_eeg_recordings,
    ensure_derivatives_dataset,
    parse_bids_entities,
    subject_derivative_path,
    validate_bids_dataset,
    validate_derivatives_dataset,
    write_json,
)


def _make_bids_recording(tmp_path: Path, *, subject: str = "01", task: str = "oddball", run: str = "01") -> Path:
    bids_root = tmp_path / "bids"
    write_json(
        bids_root / "dataset_description.json",
        {"Name": "Fixture", "BIDSVersion": "1.11.1"},
    )
    eeg_dir = bids_root / f"sub-{subject}" / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    raw_path = eeg_dir / f"sub-{subject}_task-{task}_run-{run}_eeg.vhdr"
    raw_path.write_text("MarkerFile=sub.vmrk\nDataFile=sub.eeg\n", encoding="utf-8")
    raw_path.with_suffix(".vmrk").write_text("dummy", encoding="utf-8")
    raw_path.with_suffix(".eeg").write_text("dummy", encoding="utf-8")
    (eeg_dir / f"sub-{subject}_task-{task}_run-{run}_events.tsv").write_text(
        "onset\tduration\tsample\ttrial_type\tvalue\n0.0\t0.1\t100\tStandard\t1\n",
        encoding="utf-8",
    )
    return raw_path


def test_discover_bids_eeg_recordings_and_filters(tmp_path: Path):
    raw_path = _make_bids_recording(tmp_path)
    bids_root = raw_path.parents[2]

    recordings = discover_bids_eeg_recordings(bids_root)

    assert len(recordings) == 1
    assert recordings[0].raw_path == raw_path
    assert recordings[0].events_path.name == "sub-01_task-oddball_run-01_events.tsv"
    assert discover_bids_eeg_recordings(bids_root, subjects=["sub-99"]) == []


def test_parse_bids_entities_reads_filename_and_directory_entities(tmp_path: Path):
    raw_path = _make_bids_recording(tmp_path, subject="02", task="mmn", run="03")

    entities = parse_bids_entities(raw_path)

    assert entities == {"sub": "02", "task": "mmn", "run": "03"}


def test_ensure_derivatives_dataset_and_validation_cover_sidecars(tmp_path: Path):
    raw_path = _make_bids_recording(tmp_path)
    bids_root = raw_path.parents[2]
    derivatives_root = tmp_path / "derivatives"
    dataset_root = ensure_derivatives_dataset(derivatives_root, source_dataset=bids_root)
    recording = BIDSRecording(bids_root=bids_root, raw_path=raw_path, entities=parse_bids_entities(raw_path))

    derivative_path = subject_derivative_path(
        dataset_root,
        recording.entities,
        suffix="epo",
        extension=".fif",
    )
    derivative_path.parent.mkdir(parents=True, exist_ok=True)
    derivative_path.write_text("epo", encoding="utf-8")
    derivative_path.with_suffix(".json").write_text(json.dumps({"Description": "epochs"}), encoding="utf-8")

    assert validate_bids_dataset(bids_root) == []
    assert validate_derivatives_dataset(dataset_root) == []
