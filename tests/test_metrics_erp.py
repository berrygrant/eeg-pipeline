import numpy as np
import pytest

from eeg_pipeline.metrics.erp import (
    _compute_peak,
    _get_evoked,
    _safe_pick_channels,
    compute_erp_metrics,
)
from eeg_pipeline.metrics.erp_windows import ERP_WINDOWS, ERPWindow


def test_safe_pick_channels_and_windows_cover_expected_defaults(synthetic_epochs):
    assert _safe_pick_channels(synthetic_epochs, ["Fz", "Missing"]) == ["Fz"]
    assert ERP_WINDOWS["MMN"].polarity == "negative"
    assert ERP_WINDOWS["P300"].polarity == "positive"

    with pytest.raises(ValueError, match="None of the requested channels"):
        _safe_pick_channels(synthetic_epochs, ["Missing"])


def test_get_evoked_and_compute_peak_handle_expected_cases(synthetic_epochs):
    assert _get_evoked(synthetic_epochs, "Missing") is None
    evoked = _get_evoked(synthetic_epochs, "Standard")
    assert evoked is not None

    data = np.array([1.0, -3.0, 2.0])
    times = np.array([0.1, 0.2, 0.3])
    assert _compute_peak(data, times, "negative") == (-3.0, 0.2)
    assert _compute_peak(data, times, "positive") == (2.0, 0.3)
    assert _compute_peak(data, times, "absolute") == (-3.0, 0.2)


def test_compute_erp_metrics_returns_condition_and_difference_rows(synthetic_epochs):
    window = ERPWindow(name="Test", tmin=-0.05, tmax=0.05, polarity="absolute")

    df = compute_erp_metrics(
        synthetic_epochs,
        subject="001",
        channels=["Fz"],
        windows=[window],
        compute_mmn=True,
    )

    assert set(df["condition"]) == {"Standard", "Deviant", "DEV_MINUS_STD"}
    assert len(df) == 3

    standard_row = df[df["condition"] == "Standard"].iloc[0]
    deviant_row = df[df["condition"] == "Deviant"].iloc[0]
    diff_row = df[df["condition"] == "DEV_MINUS_STD"].iloc[0]

    assert standard_row["n_epochs"] == 2
    assert deviant_row["n_epochs"] == 2
    assert diff_row["n_epochs"] == len(synthetic_epochs)
    assert diff_row["source_conditions"] == "Deviant-Standard"
    assert np.isfinite(diff_row["peak_uV"])
    assert np.isfinite(diff_row["peak_latency_s"])


def test_compute_erp_metrics_can_skip_difference_wave(synthetic_epochs):
    window = ERPWindow(name="Test", tmin=-0.05, tmax=0.05, polarity="positive")

    df = compute_erp_metrics(
        synthetic_epochs,
        subject="001",
        channels=["Fz", "Cz"],
        windows=[window],
        compute_mmn=False,
    )

    assert set(df["condition"]) == {"Standard", "Deviant"}
    assert len(df) == 4
