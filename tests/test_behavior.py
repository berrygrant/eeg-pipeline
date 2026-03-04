from pathlib import Path

import numpy as np
import pytest

from eeg_pipeline.behavior import filter_codes, read_eventcodes_from_subject_csv


def test_read_eventcodes_from_subject_csv_casts_numeric_values(tmp_path: Path):
    csv_path = tmp_path / "subject.csv"
    csv_path.write_text("EventCode\n101.0\n102.0\n", encoding="utf-8")

    codes = read_eventcodes_from_subject_csv(csv_path)

    assert np.issubdtype(codes.dtype, np.integer)
    assert np.array_equal(codes, np.array([101, 102], dtype=int))


def test_read_eventcodes_from_subject_csv_preserves_integer_dtype(tmp_path: Path):
    csv_path = tmp_path / "subject.csv"
    csv_path.write_text("EventCode\n101\n102\n", encoding="utf-8")

    codes = read_eventcodes_from_subject_csv(csv_path)

    assert np.issubdtype(codes.dtype, np.integer)
    assert np.array_equal(codes, np.array([101, 102], dtype=int))


def test_read_eventcodes_from_subject_csv_requires_eventcode_column(tmp_path: Path):
    csv_path = tmp_path / "subject.csv"
    csv_path.write_text("TrialCode\n101\n", encoding="utf-8")

    with pytest.raises(ValueError, match="'EventCode' column not found"):
        read_eventcodes_from_subject_csv(csv_path)


def test_filter_codes_returns_original_when_no_keep_list():
    codes = np.array([101, 102, 103], dtype=int)

    assert np.array_equal(filter_codes(codes, None), codes)
    assert np.array_equal(filter_codes(codes, []), codes)


def test_filter_codes_filters_to_requested_event_codes():
    codes = np.array([101, 102, 103, 101], dtype=int)

    filtered = filter_codes(codes, [101, 103])

    assert np.array_equal(filtered, np.array([101, 103, 101], dtype=int))
