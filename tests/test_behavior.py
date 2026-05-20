from pathlib import Path

import numpy as np
import pytest

from eeg_pipeline.behavior import (
    clean_eventcodes,
    filter_codes,
    read_eventcodes_from_subject_csv,
    resolve_subject_csv_path,
    write_eventcodes_csv,
)


def test_read_eventcodes_from_subject_csv_casts_numeric_values(tmp_path: Path):
    csv_path = tmp_path / "subject.csv"
    csv_path.write_text("EventCode\n101.0\n102.0\n", encoding="utf-8")

    codes = read_eventcodes_from_subject_csv(csv_path)

    assert np.issubdtype(codes.dtype, np.integer)
    assert np.array_equal(codes, np.array([101, 102], dtype=int))


def test_read_eventcodes_from_subject_csv_requires_eventcode_column(tmp_path: Path):
    csv_path = tmp_path / "subject.csv"
    csv_path.write_text("TrialCode\n101\n", encoding="utf-8")

    with pytest.raises(ValueError, match="'EventCode' column not found"):
        read_eventcodes_from_subject_csv(csv_path)


def test_resolve_subject_csv_path_supports_maria_eventcode_filenames(tmp_path: Path):
    csv_path = tmp_path / "S102-eventcodes.csv"
    csv_path.write_text("EventCode\n900\n", encoding="utf-8")

    resolved = resolve_subject_csv_path(tmp_path, "102", "S102")

    assert resolved == csv_path


def test_write_eventcodes_csv_writes_single_eventcode_column(tmp_path: Path):
    subject_csv = tmp_path / "subject-102.csv"
    subject_csv.write_text("EventCode,Other\n900.0,x\n910.0,y\n", encoding="utf-8")
    out_csv = tmp_path / "S102-eventcodes.csv"

    n_codes = write_eventcodes_csv(subject_csv, out_csv)

    assert n_codes == 2
    assert out_csv.read_text(encoding="utf-8") == "EventCode\n900\n910\n"


def test_filter_codes_returns_original_when_no_keep_list():
    codes = np.array([101, 102, 103], dtype=int)

    assert np.array_equal(filter_codes(codes, None), codes)
    assert np.array_equal(filter_codes(codes, []), codes)


def test_filter_codes_filters_to_requested_event_codes():
    codes = np.array([101, 102, 103, 101], dtype=int)

    filtered = filter_codes(codes, [101, 103])

    assert np.array_equal(filtered, np.array([101, 103, 101], dtype=int))


def test_clean_eventcodes_mprocacc_thesis_drops_first_main_buffer_event_only():
    codes = np.array([
        900, 900, 900, 910, 910, 910, 911, 910,  # practice full: keep
        800, 800, 800, 810, 810, 810, 810, 810, 810,  # practice reduced: keep
        300, 300, 300, 311, 310, 310, 310, 310, 310,  # reduced main: drop first 300
        100, 100, 100, 110, 110, 111,  # full main: drop first 100
    ], dtype=int)

    cleaned, diag = clean_eventcodes(codes, mode="mprocacc_thesis")

    assert np.array_equal(
        cleaned,
        np.array([
            900, 900, 900, 910, 910, 910, 911, 910,
            800, 800, 800, 810, 810, 810, 810, 810, 810,
            300, 300, 311, 310, 310, 310, 310, 310,
            100, 100, 110, 110, 111,
        ], dtype=int),
    )
    assert diag == {
        "eventcode_cleanup_mode": "mprocacc_thesis",
        "eventcode_cleanup_removed": 2,
        "eventcode_cleanup_runs": 2,
    }


def test_clean_eventcodes_mprocacc_thesis_requires_triplet_main_buffers():
    codes = np.array([300, 300, 310, 310, 310], dtype=int)

    with pytest.raises(ValueError, match="run_len=2"):
        clean_eventcodes(codes, mode="mprocacc_thesis")
