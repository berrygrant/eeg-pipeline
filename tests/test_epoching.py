import numpy as np
import pytest

from eeg_pipeline.epoching import (
    EpochParams,
    build_events_from_positions_and_codes,
    make_epochs,
    select_and_filter_conditions,
    select_and_recode_stddev,
)


def test_build_events_from_positions_and_codes_builds_mne_event_matrix():
    events = build_events_from_positions_and_codes(
        markers_pos=np.array([100, 200], dtype=int),
        codes=np.array([110, 111], dtype=int),
    )

    assert np.array_equal(
        events,
        np.array(
            [
                [100, 0, 110],
                [200, 0, 111],
            ],
            dtype=int,
        ),
    )


def test_build_events_from_positions_and_codes_requires_matching_lengths():
    with pytest.raises(ValueError, match="Cannot build events"):
        build_events_from_positions_and_codes(
            markers_pos=np.array([100], dtype=int),
            codes=np.array([110, 111], dtype=int),
        )


def test_select_and_recode_stddev_filters_and_relabels_events():
    events = np.array(
        [
            [100, 0, 110],
            [200, 0, 111],
            [300, 0, 999],
        ],
        dtype=int,
    )

    filtered, event_id = select_and_recode_stddev(events, standard_codes=[110], deviant_codes=[111])

    assert np.array_equal(filtered[:, 2], np.array([1, 2], dtype=int))
    assert event_id == {"Standard": 1, "Deviant": 2}


def test_select_and_filter_conditions_validates_and_filters_condition_map():
    events = np.array(
        [
            [100, 0, 110],
            [200, 0, 111],
            [300, 0, 999],
        ],
        dtype=int,
    )

    filtered, event_id, codes_flat = select_and_filter_conditions(
        events,
        {"Standard": 110, "Deviant": 111},
    )

    assert np.array_equal(filtered[:, 2], np.array([110, 111], dtype=int))
    assert event_id == {"Standard": 110, "Deviant": 111}
    assert codes_flat == [110, 111]

    with pytest.raises(ValueError, match="single code"):
        select_and_filter_conditions(events, {"Bad": [110, 111]})

    with pytest.raises(ValueError, match="Duplicate code"):
        select_and_filter_conditions(events, {"A": 110, "B": 110})

    with pytest.raises(ValueError, match="non-empty dict"):
        select_and_filter_conditions(events, {})


def test_make_epochs_builds_epochs_from_raw_and_events(synthetic_raw):
    events = np.array(
        [
            [100, 0, 1],
            [300, 0, 2],
            [500, 0, 1],
        ],
        dtype=int,
    )

    epochs = make_epochs(
        synthetic_raw,
        events_stddev=events,
        event_id={"Standard": 1, "Deviant": 2},
        ep=EpochParams(tmin=-0.1, tmax=0.2, baseline=(-0.1, 0.0)),
    )

    assert len(epochs) == 3
    assert epochs.event_id == {"Standard": 1, "Deviant": 2}
