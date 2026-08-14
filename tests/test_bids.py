import json
from pathlib import Path

from eeg_pipeline.bids import (
    BIDSRecording,
    discover_bids_eeg_recordings,
    ensure_derivatives_dataset,
    filter_derivative_paths,
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


def _derivative_epochs_paths() -> list[Path]:
    return [
        Path("sub-01_task-oddball_run-01_desc-clean_epo.fif"),
        Path("sub-01_task-oddball_run-02_desc-clean_epo.fif"),
        Path("sub-02_task-oddball_run-01_desc-clean_epo.fif"),
        Path("sub-03_ses-02_task-mmn_run-01_desc-clean_epo.fif"),
    ]


def test_filter_derivative_paths_without_filters_is_identity():
    paths = _derivative_epochs_paths()

    assert filter_derivative_paths(paths) == paths


def test_filter_derivative_paths_selects_single_subject():
    paths = _derivative_epochs_paths()

    # Bare and prefixed forms must behave identically, matching the semantics of
    # discover_bids_eeg_recordings, so an array job can pass either.
    for value in ("01", "sub-01"):
        kept = filter_derivative_paths(paths, subjects=[value])
        assert [p.name for p in kept] == [
            "sub-01_task-oddball_run-01_desc-clean_epo.fif",
            "sub-01_task-oddball_run-02_desc-clean_epo.fif",
        ]


def test_filter_derivative_paths_applies_session_task_and_run_filters():
    paths = _derivative_epochs_paths()

    assert [p.name for p in filter_derivative_paths(paths, sessions=["02"])] == [
        "sub-03_ses-02_task-mmn_run-01_desc-clean_epo.fif"
    ]
    assert [p.name for p in filter_derivative_paths(paths, tasks=["mmn"])] == [
        "sub-03_ses-02_task-mmn_run-01_desc-clean_epo.fif"
    ]
    assert [p.name for p in filter_derivative_paths(paths, runs=["02"])] == [
        "sub-01_task-oddball_run-02_desc-clean_epo.fif"
    ]
    # Filters compose (AND), so a subject+run pair narrows to one file.
    assert [p.name for p in filter_derivative_paths(paths, subjects=["01"], runs=["01"])] == [
        "sub-01_task-oddball_run-01_desc-clean_epo.fif"
    ]


def test_filter_derivative_paths_drops_non_bids_names_when_filtering():
    paths = [*_derivative_epochs_paths(), Path("grand-average_epo.fif")]

    # A name carrying no sub- entity cannot be confirmed to match, so it is
    # excluded rather than silently passed through to a per-subject job.
    kept = filter_derivative_paths(paths, subjects=["01"])
    assert all("sub-01" in p.name for p in kept)
    assert len(kept) == 2


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_discovery_skips_bids_reserved_directories(tmp_path: Path):
    """sourcedata/, derivatives/ and VCS internals must not be scanned.

    ds003620 keeps its non-BIDS originals in sourcedata/sub-1/eeg/runabout1.vhdr;
    scanning that raised and aborted discovery for the whole dataset. Scanning
    derivatives/ is worse in a quieter way: it feeds the pipeline's own outputs
    back in as though they were raw input.
    """
    root = tmp_path / "ds"
    _touch(root / "sub-01/eeg/sub-01_task-oddball_run-01_eeg.vhdr")
    _touch(root / "sourcedata/sub-1/eeg/runabout1.vhdr")
    _touch(root / "derivatives/eeg-pipeline/sub-01/eeg/sub-01_task-oddball_run-01_desc-preproc_eeg.vhdr")
    _touch(root / "code/scratch_eeg.vhdr")
    _touch(root / "stimuli/tone_eeg.vhdr")
    _touch(root / ".datalad/weird.vhdr")

    recs = discover_bids_eeg_recordings(root)

    assert [r.relative_raw_path for r in recs] == [
        "sub-01/eeg/sub-01_task-oddball_run-01_eeg.vhdr"
    ]


def test_discovery_reports_unparseable_names_without_aborting(tmp_path: Path, capsys):
    """One oddly named file must not cost the entire dataset.

    It is still reported: inside the BIDS tree proper a non-BIDS name may be a
    recording that should have been processed, and dropping it silently is the
    failure this whole exercise keeps running into.
    """
    root = tmp_path / "ds"
    _touch(root / "sub-01/eeg/sub-01_task-oddball_run-01_eeg.vhdr")
    _touch(root / "sub-02/eeg/sub-02_task-oddball_run-01_eeg.vhdr")
    _touch(root / "sub-01/eeg/stray_notes.vhdr")

    recs = discover_bids_eeg_recordings(root)

    assert len(recs) == 2
    out = capsys.readouterr().out
    assert "stray_notes.vhdr" in out
    assert "non-BIDS names" in out
