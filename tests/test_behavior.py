from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.behavior import (
    behavior_keep_mask,
    extract_codes_from_bids_events,
    filter_codes,
    load_behavioral_events,
    read_eventcodes_from_subject_csv,
)


def test_read_eventcodes_from_subject_csv_casts_numeric_values(tmp_path: Path):
    csv_path = tmp_path / "subject.csv"
    csv_path.write_text("EventCode\n101.0\n102.0\n", encoding="utf-8")

    codes = read_eventcodes_from_subject_csv(csv_path)

    assert np.issubdtype(codes.dtype, np.integer)
    assert np.array_equal(codes, np.array([101, 102], dtype=int))


def test_filter_codes_and_keep_mask_follow_requested_values():
    codes = np.array([101, 102, 103, 101], dtype=int)

    assert np.array_equal(filter_codes(codes, [101, 103]), np.array([101, 103, 101], dtype=int))
    assert np.array_equal(behavior_keep_mask(codes, [101, 103]), np.array([True, False, True, True]))


def test_extract_codes_from_bids_events_prefers_numeric_value_column():
    events_df = pd.DataFrame({"onset": [0.0, 1.0], "value": ["11", "12"]})

    codes = extract_codes_from_bids_events(events_df)

    assert np.array_equal(codes, np.array([11, 12], dtype=int))


def test_extract_codes_from_bids_events_can_map_trial_type_with_condition_map():
    events_df = pd.DataFrame({"onset": [0.0, 1.0], "trial_type": ["Standard", "Deviant"]})

    codes = extract_codes_from_bids_events(events_df, condition_map={"Standard": [1], "Deviant": [2]})

    assert np.array_equal(codes, np.array([1, 2], dtype=int))


def test_load_behavioral_events_prefers_bids_events_and_preserves_samples(tmp_path: Path):
    events_tsv = tmp_path / "sub-01_task-oddball_events.tsv"
    events_json = tmp_path / "sub-01_task-oddball_events.json"
    events_tsv.write_text(
        "onset\tduration\tsample\ttrial_type\tvalue\n0.0\t0.1\t100\tStandard\t1\n1.0\t0.1\t200\tDeviant\t2\n",
        encoding="utf-8",
    )
    events_json.write_text('{"trial_type":{"Description":"Condition"}}', encoding="utf-8")

    loaded = load_behavioral_events(
        events_tsv=events_tsv,
        events_json=events_json,
        subject_id="01",
        keep_codes=[2],
        token_map={"token1": "A", "token2": "B"},
        condition_map=None,
        csv_fallback_dir=None,
    )

    assert loaded.source == "bids_events"
    assert loaded.source_path == events_tsv
    assert np.array_equal(loaded.codes_all, np.array([1, 2], dtype=int))
    assert np.array_equal(loaded.codes, np.array([2], dtype=int))
    assert np.array_equal(loaded.samples, np.array([200], dtype=int))
    assert loaded.metadata.iloc[0]["trial_type"] == "Deviant"


def test_load_behavioral_events_uses_explicit_csv_fallback_when_events_missing(tmp_path: Path):
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    (fallback_dir / "subject-01.csv").write_text("EventCode\n1\n2\n", encoding="utf-8")

    loaded = load_behavioral_events(
        events_tsv=tmp_path / "missing_events.tsv",
        events_json=tmp_path / "missing_events.json",
        subject_id="01",
        keep_codes=None,
        token_map={"token1": "A", "token2": "B"},
        condition_map=None,
        csv_fallback_dir=fallback_dir,
    )

    assert loaded.source == "csv_fallback"
    assert loaded.source_path == fallback_dir / "subject-01.csv"
    assert np.array_equal(loaded.codes_all, np.array([1, 2], dtype=int))
    assert loaded.samples is None


def test_load_behavioral_events_requires_bids_events_or_explicit_fallback(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Missing BIDS events file"):
        load_behavioral_events(
            events_tsv=tmp_path / "missing_events.tsv",
            events_json=tmp_path / "missing_events.json",
            subject_id="01",
            keep_codes=None,
            token_map=None,
            condition_map=None,
            csv_fallback_dir=None,
        )
