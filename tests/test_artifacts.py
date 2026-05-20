import numpy as np
import pytest

from eeg_pipeline.artifacts import (
    moving_window_ptp_mask,
    moving_window_ptp_max,
    simple_voltage_threshold_mask,
    step_threshold_mask,
)


def test_moving_window_ptp_mask_uses_full_epoch_when_window_exceeds_data():
    data = np.array(
        [
            [0.0, 50e-6, 0.0, 0.0],
            [0.0, 10e-6, 0.0, 0.0],
        ]
    )

    bad = moving_window_ptp_mask(
        data_v=data,
        sfreq=1000.0,
        win_ms=10.0,
        step_ms=1.0,
        threshold_uv=40.0,
    )

    assert np.array_equal(bad, np.array([True, False]))


def test_moving_window_ptp_mask_and_max_handle_sliding_windows():
    data = np.array(
        [
            [[0.0, 100e-6, 0.0, 0.0, 0.0]],
            [[0.0, 20e-6, 0.0, 0.0, 0.0]],
        ]
    )

    bad = moving_window_ptp_mask(
        data_v=data,
        sfreq=1000.0,
        win_ms=2.0,
        step_ms=1.0,
        threshold_uv=50.0,
    )
    max_ptp = moving_window_ptp_max(
        data_v=data,
        sfreq=1000.0,
        win_ms=2.0,
        step_ms=1.0,
    )

    assert np.array_equal(bad, np.array([True, False]))
    assert np.allclose(max_ptp, np.array([100.0, 20.0]))


def test_simple_voltage_threshold_mask_flags_positive_and_negative_limits():
    data = np.array(
        [
            [0.0, 200e-6, 0.0],
            [0.0, -200e-6, 0.0],
            [0.0, 20e-6, 0.0],
        ]
    )

    bad = simple_voltage_threshold_mask(data_v=data, pos_limit_uv=150.0, neg_limit_uv=-150.0)

    assert np.array_equal(bad, np.array([True, True, False]))


def test_step_threshold_mask_flags_large_steps_and_rejects_nonpositive_sfreq():
    data = np.array(
        [
            [0.0, 0.0, 100e-6],
            [0.0, 10e-6, 20e-6],
        ]
    )

    bad = step_threshold_mask(data_v=data, sfreq=1000.0, threshold_uv_per_ms=50.0)

    assert np.array_equal(bad, np.array([True, False]))

    with pytest.raises(ValueError, match="sfreq must be positive"):
        step_threshold_mask(data_v=data, sfreq=0.0, threshold_uv_per_ms=50.0)
