import numpy as np
import pytest

import eeg_pipeline.artifacts as artifacts
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


def test_moving_window_ptp_max_uses_full_epoch_when_window_exceeds_data():
    data = np.array(
        [
            [0.0, 80e-6, 0.0, 0.0],
            [0.0, 30e-6, 0.0, 0.0],
        ]
    )

    max_ptp = moving_window_ptp_max(
        data_v=data,
        sfreq=1000.0,
        win_ms=10.0,
        step_ms=1.0,
    )

    assert np.allclose(max_ptp, np.array([80.0, 30.0]))


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


def test_artifact_masks_reject_invalid_ndim_inputs():
    bad = np.zeros((1, 1, 1, 1), dtype=float)

    with pytest.raises(ValueError, match="Expected 2D or 3D array"):
        moving_window_ptp_mask(bad, sfreq=1000.0, win_ms=2.0, step_ms=1.0, threshold_uv=1.0)
    with pytest.raises(ValueError, match="Expected 2D or 3D array"):
        moving_window_ptp_max(bad, sfreq=1000.0, win_ms=2.0, step_ms=1.0)
    with pytest.raises(ValueError, match="Expected 2D or 3D array"):
        simple_voltage_threshold_mask(bad, pos_limit_uv=1.0, neg_limit_uv=-1.0)
    with pytest.raises(ValueError, match="Expected 2D or 3D array"):
        step_threshold_mask(bad, sfreq=1000.0, threshold_uv_per_ms=1.0)


def test_maybe_to_numpy_uses_gpu_conversion_when_backend_is_cupy(monkeypatch):
    marker = object()

    monkeypatch.setattr(artifacts, "gpu_backend", lambda: "cupy")
    monkeypatch.setattr(artifacts, "to_numpy", lambda x: ["converted", x])

    assert artifacts._maybe_to_numpy(marker) == ["converted", marker]
