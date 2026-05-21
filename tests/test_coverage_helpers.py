from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline import align, behavior, bids, inputs, io_brainvision
from eeg_pipeline.schema import SchemaV1Config


def test_behavior_legacy_csv_helpers_cover_candidate_and_error_paths(tmp_path: Path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    preferred = csv_dir / "S123-eventcodes.csv"
    preferred.write_text("EventCode\n1\n2\n", encoding="utf-8")

    assert behavior.subject_number_from_stem("S123") == "123"
    assert behavior.subject_number_from_stem("subject_123") == "123"
    with pytest.raises(ValueError, match="Cannot parse"):
        behavior.subject_number_from_stem("subject")

    assert behavior.resolve_subject_csv_path(csv_dir, "123", "S123") == preferred
    assert behavior.resolve_subject_csv_path(csv_dir, "999") == csv_dir / "subject-999.csv"

    out_csv = tmp_path / "out" / "events.csv"
    assert behavior.write_eventcodes_csv(preferred, out_csv) == 2
    assert out_csv.exists()

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("Other\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="EventCode"):
        behavior.read_eventcodes_from_subject_csv(bad_csv)


def test_behavior_event_cleanup_and_bids_code_derivation_edges():
    unchanged, unchanged_diag = behavior.clean_eventcodes([1, 2], mode=None)
    assert unchanged.tolist() == [1, 2]
    assert unchanged_diag["eventcode_cleanup_mode"] == "none"

    cfg = SchemaV1Config(full_A=[1], reduced_A=[], practice_A=[])
    cleaned, diag = behavior.clean_eventcodes(
        np.array([100, 100, 100, 111, 111]),
        mode="mprocacc_thesis",
        cfg=cfg,
    )
    assert cleaned.tolist() == [100, 100, 111, 111]
    assert diag["eventcode_cleanup_removed"] == 1
    assert diag["eventcode_cleanup_runs"] == 1

    empty, empty_diag = behavior.clean_eventcodes([], mode="mprocacc_thesis", cfg=cfg)
    assert empty.tolist() == []
    assert empty_diag["eventcode_cleanup_removed"] == 0

    with pytest.raises(ValueError, match="Unsupported"):
        behavior.clean_eventcodes([1], mode="unknown")
    with pytest.raises(ValueError, match="expected main-block"):
        behavior.clean_eventcodes(np.array([100, 100]), mode="mprocacc_thesis", cfg=cfg)

    trial_type_df = pd.DataFrame({"trial_type": ["Standard", "Deviant"]})
    assert behavior.extract_codes_from_bids_events(
        trial_type_df,
        condition_map={"Standard": [1], "Deviant": [2]},
    ).tolist() == [1, 2]
    with pytest.raises(ValueError, match="single numeric code"):
        behavior.extract_codes_from_bids_events(trial_type_df, condition_map={"Standard": [1, 2]})
    with pytest.raises(ValueError, match="Could not derive"):
        behavior.extract_codes_from_bids_events(trial_type_df, condition_map={"Other": [1]})
    with pytest.raises(ValueError, match="BIDS events must include"):
        behavior.extract_codes_from_bids_events(pd.DataFrame({"trial_type": ["Standard"]}))

    codes = np.array([1, 2, 3])
    assert behavior.filter_codes(codes, None) is codes
    assert behavior.filter_codes(codes, []).tolist() == [1, 2, 3]
    assert behavior.behavior_keep_mask(codes, None).tolist() == [True, True, True]

    with pytest.raises(FileNotFoundError, match="BIDS events file"):
        behavior.read_bids_events_table(Path("missing_events.tsv"))


def test_load_behavioral_events_supports_bids_sidecar_samples_and_csv_fallback(tmp_path: Path):
    events_tsv = tmp_path / "sub-01_task-test_events.tsv"
    events_json = tmp_path / "sub-01_task-test_events.json"
    events_tsv.write_text(
        "onset\tduration\tsample\ttrial_type\tvalue\n"
        "0\t0\t10\tStandard\t1\n"
        "1\t0\t20\tDeviant\t2\n"
        "2\t0\tbad\tOther\t3\n",
        encoding="utf-8",
    )
    events_json.write_text('{"value":{"Description":"code"}}', encoding="utf-8")

    loaded = behavior.load_behavioral_events(
        events_tsv=events_tsv,
        events_json=events_json,
        subject_id="01",
        keep_codes=[1, 2],
        condition_map={"Standard": [1], "Deviant": [2]},
    )
    assert loaded.source == "bids_events"
    assert loaded.sidecar["value"]["Description"] == "code"
    assert loaded.codes.tolist() == [1, 2]
    assert loaded.samples is None

    sample_tsv = tmp_path / "sub-04_task-test_events.tsv"
    sample_tsv.write_text(
        "onset\tduration\tsample\ttrial_type\tvalue\n"
        "0\t0\t10\tStandard\t1\n"
        "1\t0\t20\tDeviant\t2\n",
        encoding="utf-8",
    )
    sample_loaded = behavior.load_behavioral_events(
        events_tsv=sample_tsv,
        events_json=tmp_path / "missing_events.json",
        subject_id="04",
        keep_codes=[2],
        condition_map={"Standard": [1], "Deviant": [2]},
    )
    assert sample_loaded.sidecar == {}
    assert sample_loaded.samples.tolist() == [20]

    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    fallback = fallback_dir / "sub-02.csv"
    fallback.write_text("EventCode\n1\n2\n3\n", encoding="utf-8")
    csv_loaded = behavior.load_behavioral_events(
        events_tsv=tmp_path / "missing_events.tsv",
        events_json=tmp_path / "missing_events.json",
        subject_id="02",
        keep_codes=[1, 3],
        csv_fallback_dir=fallback_dir,
        condition_map={"Standard": [1], "Other": [3]},
    )
    assert csv_loaded.source == "csv_fallback"
    assert csv_loaded.codes.tolist() == [1, 3]
    assert csv_loaded.metadata["condition"].tolist() == ["Standard", "Other"]

    with pytest.raises(FileNotFoundError, match="CSV fallback"):
        behavior.load_behavioral_events(
            events_tsv=tmp_path / "missing_events.tsv",
            events_json=tmp_path / "missing_events.json",
            subject_id="03",
            keep_codes=None,
            csv_fallback_dir=fallback_dir,
        )

    explicit_csv = fallback_dir / "explicit.csv"
    explicit_csv.write_text("EventCode\n2\n", encoding="utf-8")
    explicit_loaded = behavior.load_behavioral_events(
        events_tsv=tmp_path / "still_missing_events.tsv",
        events_json=tmp_path / "missing_events.json",
        subject_id="02",
        keep_codes=None,
        csv_path=explicit_csv,
        token_map={"token1": "EH", "token2": "IH"},
    )
    assert explicit_loaded.source_path == explicit_csv
    assert explicit_loaded.codes.tolist() == [2]

    with pytest.raises(FileNotFoundError, match="Missing BIDS events file"):
        behavior.load_behavioral_events(
            events_tsv=tmp_path / "no_events.tsv",
            events_json=tmp_path / "missing_events.json",
            subject_id="02",
            keep_codes=None,
        )


def test_bids_paths_validation_and_filter_edges(tmp_path: Path):
    bids_root = tmp_path / "bids"
    eeg_dir = bids_root / "sub-01" / "ses-02" / "eeg"
    eeg_dir.mkdir(parents=True)
    raw = eeg_dir / "sub-01_ses-02_task-odd_run-01_eeg.vhdr"
    raw.write_text("raw", encoding="utf-8")
    (eeg_dir / "sub-01_ses-02_task-odd_run-01_events.tsv").write_text("onset\tduration\n", encoding="utf-8")
    (bids_root / "dataset_description.json").write_text(
        '{"Name":"Fixture","BIDSVersion":"1.11.1"}',
        encoding="utf-8",
    )
    deriv = bids_root / "derivatives" / "sub-99" / "eeg"
    deriv.mkdir(parents=True)
    (deriv / "sub-99_task-ignore_eeg.vhdr").write_text("ignored", encoding="utf-8")

    recording = bids.BIDSRecording(bids_root, raw, bids.parse_bids_entities(raw))
    assert recording.subject_label == "sub-01"
    assert recording.session_label == "ses-02"
    assert recording.task_id == "odd"
    assert recording.run_id == "01"
    assert recording.raw_json_path == raw.with_suffix(".json")
    assert recording.relative_raw_path.endswith("_eeg.vhdr")
    assert recording.relative_events_path.endswith("_events.tsv")

    discovered = bids.discover_bids_eeg_recordings(
        bids_root,
        subjects=["sub-01"],
        sessions=["02"],
        tasks=["task-odd"],
        runs=["01"],
    )
    assert [r.raw_path for r in discovered] == [raw]
    assert bids.discover_bids_eeg_recordings(bids_root, subjects=["99"]) == []
    with pytest.raises(FileNotFoundError):
        bids.discover_bids_eeg_recordings(tmp_path / "missing")

    assert bids.build_bids_basename({"sub": "01", "task": "odd"}, suffix="epo", desc="clean") == "sub-01_task-odd_desc-clean_epo"
    dataset_root = bids.ensure_derivatives_dataset(tmp_path / "derivatives", source_dataset=bids_root, pipeline_version="1.2.3")
    subject_path = bids.subject_derivative_path(dataset_root, {"sub": "01", "ses": "02", "task": "odd"}, suffix="epo", extension="fif")
    dataset_path = bids.dataset_derivative_path(dataset_root, suffix="metrics", extension="tsv", desc="erp")
    assert subject_path.name == "sub-01_ses-02_task-odd_epo.fif"
    assert dataset_path.name == "desc-erp_metrics.tsv"
    assert bids.sidecar_json_path(dataset_path.with_suffix(".json")) == dataset_path.with_suffix(".json")
    assert bids.source_basename_from_derivative_path(subject_path) == "sub-01_ses-02_task-odd_eeg"

    assert bids.validate_bids_dataset(bids_root) == []
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "dataset_description.json").write_text("{", encoding="utf-8")
    errors = bids.validate_bids_dataset(invalid)
    assert any("Invalid JSON" in error for error in errors)
    assert any("No BIDS EEG recordings" in error for error in errors)

    no_description = tmp_path / "no_description"
    (no_description / "sub-01" / "eeg").mkdir(parents=True)
    (no_description / "sub-01" / "eeg" / "sub-01_task-odd_eeg.vhdr").write_text("raw", encoding="utf-8")
    description_errors = bids.validate_bids_dataset(no_description)
    assert any("Missing dataset_description.json" in error for error in description_errors)
    assert any("Missing events.tsv" in error for error in description_errors)

    derivative_errors = bids.validate_derivatives_dataset(dataset_root)
    assert any("No derivative files" in error for error in derivative_errors)
    dataset_path.write_text("metric\n", encoding="utf-8")
    assert any("Missing JSON sidecar" in error for error in bids.validate_derivatives_dataset(dataset_root))
    bids.write_json(bids.derivative_sidecar_path(dataset_path), {"Description": "ok"})
    assert bids.validate_derivatives_dataset(dataset_root) == []

    broken_derivatives = tmp_path / "broken_derivatives"
    broken_derivatives.mkdir()
    assert any("Missing dataset_description.json" in error for error in bids.validate_derivatives_dataset(broken_derivatives))
    (broken_derivatives / "dataset_description.json").write_text("{", encoding="utf-8")
    assert any("Invalid JSON" in error for error in bids.validate_derivatives_dataset(broken_derivatives))
    bids.write_json(
        broken_derivatives / "dataset_description.json",
        {"Name": "broken", "BIDSVersion": "1.11.1", "DatasetType": "raw"},
    )
    (broken_derivatives / "eeg").mkdir()
    (broken_derivatives / "eeg" / "desc-demo_metrics.tsv").write_text("metric\n", encoding="utf-8")
    broken_errors = bids.validate_derivatives_dataset(broken_derivatives)
    assert any("Missing GeneratedBy" in error for error in broken_errors)
    assert any("DatasetType must be 'derivative'" in error for error in broken_errors)


def test_bids_rejects_inconsistent_filenames(tmp_path: Path):
    with pytest.raises(ValueError, match="not a BIDS"):
        bids.parse_bids_entities(Path("demo.vhdr"))
    with pytest.raises(ValueError, match="Expected BIDS EEG suffix"):
        bids.parse_bids_entities(Path("sub-01_task-test_events.tsv"))
    with pytest.raises(ValueError, match="Missing sub"):
        bids.parse_bids_entities_like_name("task-test_epo")
    with pytest.raises(ValueError, match="Subject directory"):
        bids.parse_bids_entities(tmp_path / "sub-02" / "eeg" / "sub-01_task-test_eeg.vhdr")
    with pytest.raises(ValueError, match="Session directory"):
        bids.parse_bids_entities(tmp_path / "sub-01" / "ses-02" / "eeg" / "sub-01_ses-03_task-test_eeg.vhdr")


def test_alignment_helpers_cover_empty_and_burst_edge_cases():
    assert align.marker_gap_stats(np.array([10]), sfreq=100.0) == {"n": 1}

    formatted = align.format_alignment_diag(
        {
            "markers_original": 5,
            "markers_after_burst_collapse": 4,
            "markers_dropped_by_burst": 1,
            "markers_dropped_by_gap": 2,
            "markers_dropped_by_auto": 3,
        },
        aligned_n=3,
    )
    assert formatted == (
        "Alignment: markers 5 -> 4 after burst collapse -> 3 "
        "(burst_drop=1, gap_drop=2, auto_drop=3)"
    )

    burst = align.detect_trigger_bursts(np.array([10]), sfreq=100.0)
    assert burst == {
        "burst_flag": False,
        "n_triggers": 1,
        "n_short_iti": 0,
        "min_iti_s": None,
        "burst_max_in_window": 1,
        "burst_n_windows_ge_thresh": 0,
    }

    collapsed, diag = align.collapse_marker_bursts(np.array([], dtype=int), sfreq=100.0, min_iti_s=0.02)
    assert collapsed.tolist() == []
    assert diag["markers_after_burst_collapse"] == 0

    single, diag = align.collapse_marker_bursts(np.array([42]), sfreq=100.0, min_iti_s=0.02)
    assert single.tolist() == [42]
    assert diag["markers_after_burst_collapse"] == 1

    with pytest.raises(ValueError, match="Unsupported burst keep"):
        align.collapse_marker_bursts(np.array([1, 2]), sfreq=100.0, min_iti_s=0.02, keep="middle")
    with pytest.raises(ValueError, match="min_iti_s must be"):
        align.collapse_marker_bursts(np.array([1, 2]), sfreq=100.0, min_iti_s=-0.01)

    last, diag = align.collapse_marker_bursts(np.array([0, 1, 10]), sfreq=100.0, min_iti_s=0.02, keep="last")
    assert last.tolist() == [1, 10]
    assert diag["markers_dropped_by_burst"] == 1


def test_inputs_discovery_and_conversion_helpers(monkeypatch, tmp_path: Path):
    raw_dir = tmp_path / "legacy"
    csv_dir = tmp_path / "csv"
    raw_dir.mkdir()
    csv_dir.mkdir()
    vhdr = raw_dir / "sub-001_ses-01_task-odd_run-02_eeg.vhdr"
    vmrk = vhdr.with_suffix(".vmrk")
    eeg = vhdr.with_suffix(".eeg")
    vhdr.write_text(
        "MarkerFile=sub-001_ses-01_task-odd_run-02_eeg.vmrk\n"
        "DataFile=sub-001_ses-01_task-odd_run-02_eeg.eeg\n",
        encoding="utf-8",
    )
    vmrk.write_text("markers", encoding="utf-8")
    eeg.write_text("data", encoding="utf-8")
    (csv_dir / "subject-001.csv").write_text("EventCode\n1\n2\n", encoding="utf-8")

    recording = inputs.discover_legacy_recordings(
        raw_dir,
        subject_csv_dir=csv_dir,
        subjects=["sub-001"],
        sessions=["01"],
        runs=["02"],
        task_label="odd",
    )[0]
    assert recording.subject_label == "sub-001"
    assert recording.session_label == "ses-01"
    assert recording.relative_behavior_path.endswith("csv/subject-001.csv")
    assert inputs.discover_legacy_recordings(raw_dir, subject_csv_dir=csv_dir, subjects=["999"]) == []
    with pytest.raises(FileNotFoundError):
        inputs.discover_legacy_recordings(tmp_path / "missing", subject_csv_dir=None)
    with pytest.raises(ValueError, match="Legacy mode requires"):
        inputs.discover_pipeline_recordings(mode="legacy", bids_root=None, raw_dir=None, subject_csv_dir=None)
    with pytest.raises(ValueError, match="BIDS mode requires"):
        inputs.discover_pipeline_recordings(mode="bids", bids_root=None, raw_dir=None, subject_csv_dir=None)
    assert inputs.subject_number_from_stem("subject_001") == "001"
    with pytest.raises(ValueError):
        inputs.subject_number_from_stem("subject")

    fake_raw = SimpleNamespace(
        info={"sfreq": 100.0},
        ch_names=["Fz", "EOG"],
        get_channel_types=lambda: ["eeg", "eog"],
    )
    monkeypatch.setattr(inputs, "_read_raw_minimal", lambda path: fake_raw)
    monkeypatch.setattr(inputs, "events_from_annotations_positions", lambda raw: np.array([[10, 0, 1], [20, 0, 2]]))
    monkeypatch.setattr(inputs, "align_marker_positions_to_codes", lambda **kwargs: (np.array([10, 20]), {}))

    converted = inputs.convert_legacy_recordings_to_bids(
        [recording],
        bids_root=tmp_path / "converted",
        task_label="odd",
        keep_codes=[1, 2],
        standard_codes=[1],
        deviant_codes=[2],
        drop_eeg_markers_by_gap_s=None,
        auto_drop_to_count=True,
    )
    converted_recording = converted[0]
    assert converted_recording.source_type == "bids"
    assert converted_recording.behavior_kind == "bids_events"
    assert converted_recording.behavior_path.exists()
    assert "sub-001" in (tmp_path / "converted" / "participants.tsv").read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        inputs.convert_legacy_recordings_to_bids(
            [recording],
            bids_root=tmp_path / "converted",
            task_label="odd",
            keep_codes=[1, 2],
            standard_codes=[1],
            deviant_codes=[2],
            drop_eeg_markers_by_gap_s=None,
            auto_drop_to_count=True,
            overwrite=False,
        )


def test_inputs_copy_set_sidecar_and_raw_reader_errors(tmp_path: Path):
    set_path = tmp_path / "S002.set"
    fdt_path = tmp_path / "S002.fdt"
    set_path.write_text("set", encoding="utf-8")
    fdt_path.write_text("fdt", encoding="utf-8")
    recording = inputs.PipelineRecording(
        source_type="legacy",
        source_root=tmp_path,
        raw_path=set_path,
        entities={"sub": "002"},
    )
    target = tmp_path / "bids" / "sub-002" / "eeg" / "sub-002_task-test_eeg.set"
    inputs._copy_legacy_raw_to_bids(recording, target)
    assert target.exists()
    assert target.with_suffix(".fdt").exists()
    assert (target.parent / "S002.fdt").exists()

    with pytest.raises(ValueError, match="Unsupported raw"):
        inputs._read_raw_minimal(tmp_path / "demo.txt")
    assert inputs._events_sidecar()["sample"]["Description"]


def test_io_brainvision_helpers_cover_link_and_marker_edges(tmp_path: Path):
    vhdr = tmp_path / "demo.vhdr"
    vhdr.write_text("MarkerFile=demo.vmrk\nDataFile=demo.eeg\n", encoding="utf-8")
    ok, reason = io_brainvision.brainvision_links_ok(vhdr)
    assert ok is False
    assert "MarkerFile=demo.vmrk" in reason
    assert io_brainvision._bv_get(vhdr.read_text(encoding="utf-8"), "datafile") == "demo.eeg"
    assert io_brainvision._bv_get("NoKey=value", "missing") is None

    vmrk = tmp_path / "demo.vmrk"
    vmrk.write_text(
        "Mk1=Stimulus,S 1,100,1,0\n"
        "Comment=ignored\n"
        "Mk2=Too,Short\n",
        encoding="utf-8",
    )
    markers = io_brainvision.parse_vmrk_markers(vmrk)
    assert markers.to_dict("records") == [
        {"mk": 1, "mtype": "Stimulus", "desc": "S 1", "pos": 100, "size": 1, "chan": 0}
    ]

    with pytest.raises(ValueError, match="Unsupported raw"):
        io_brainvision.read_raw_preprocess(
            tmp_path / "demo.txt",
            montage="standard_1020",
            eog_chs=[],
            aux_chs=[],
            reref="average",
            l_freq=0.1,
            h_freq=30.0,
            notch=None,
        )
