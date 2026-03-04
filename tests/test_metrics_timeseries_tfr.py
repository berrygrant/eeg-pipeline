from types import SimpleNamespace

import numpy as np
import pytest

import eeg_pipeline.metrics.erp_timeseries as erp_ts_mod
import eeg_pipeline.metrics.tfr as tfr_mod


class FakeTFR:
    def __init__(self, data, times, freqs, ch_names):
        self.data = np.array(data, dtype=float)
        self.times = np.array(times, dtype=float)
        self.freqs = np.array(freqs, dtype=float)
        self.ch_names = list(ch_names)

    def copy(self):
        return FakeTFR(self.data.copy(), self.times.copy(), self.freqs.copy(), list(self.ch_names))


def test_erp_timeseries_safe_pick_channels_and_long_df(synthetic_epochs):
    ev = synthetic_epochs["Standard"].average().pick(["Fz"])

    assert erp_ts_mod._safe_pick_channels(synthetic_epochs, ["Fz", "Missing"]) == ["Fz"]

    df = erp_ts_mod._evoked_to_long_df(
        ev,
        subject="001",
        condition="Standard",
        n_epochs=2,
        decim=2,
    )

    assert set(df.columns) == {
        "subject",
        "condition",
        "n_epochs",
        "channel",
        "time_s",
        "amplitude_uv",
    }
    assert df["subject"].nunique() == 1
    assert df["condition"].nunique() == 1
    assert df["channel"].unique().tolist() == ["Fz"]
    assert len(df) == len(ev.copy().decimate(2, offset=0).times)


def test_compute_erp_timeseries_handles_empty_and_missing_conditions(synthetic_epochs):
    empty_df = erp_ts_mod.compute_erp_timeseries(
        synthetic_epochs[:0],
        subject="001",
        channels=["Fz"],
    )
    assert empty_df.iloc[0]["status"] == "EMPTY_EPOCHS"

    missing_df = erp_ts_mod.compute_erp_timeseries(
        synthetic_epochs,
        subject="001",
        channels=["Fz"],
        conditions=["Missing"],
        params=erp_ts_mod.ERPTimeSeriesParams(tmin=-0.1, tmax=0.2, baseline=None),
    )
    assert missing_df.iloc[0]["status"] == "NO_CONDITIONS"


def test_compute_erp_timeseries_returns_conditions_and_difference_wave(synthetic_epochs):
    params = erp_ts_mod.ERPTimeSeriesParams(tmin=-0.05, tmax=0.05, baseline=None, decim=2)

    df = erp_ts_mod.compute_erp_timeseries(
        synthetic_epochs,
        subject="001",
        channels=["Fz"],
        params=params,
        include_difference_wave=True,
    )

    assert set(df["condition"]) == {"Standard", "Deviant", "DEV_MINUS_STD"}
    assert set(df["status"]) == {"OK"}
    diff = df[df["condition"] == "DEV_MINUS_STD"]
    assert diff["n_epochs"].iloc[0] == 2
    assert diff["channel"].unique().tolist() == ["Fz"]


def test_compute_erp_timeseries_applies_baseline_and_validates_channels(synthetic_epochs):
    params = erp_ts_mod.ERPTimeSeriesParams(tmin=-0.1, tmax=0.2, baseline=(-0.1, 0.0), decim=1)

    df = erp_ts_mod.compute_erp_timeseries(
        synthetic_epochs,
        subject="001",
        channels=["Fz"],
        params=params,
        include_difference_wave=False,
    )
    assert set(df["status"]) == {"OK"}

    with pytest.raises(ValueError, match="None of requested channels found"):
        erp_ts_mod.compute_erp_timeseries(
            synthetic_epochs,
            subject="001",
            channels=["Missing"],
            params=params,
            include_difference_wave=False,
        )


def test_compute_erp_timeseries_skips_conditions_with_zero_epochs():
    class EmptyConditionEpochs:
        ch_names = ["Fz"]
        event_id = {"Empty": 1}

        def __len__(self):
            return 1

        def copy(self):
            return self

        def pick(self, chs):
            assert chs == ["Fz"]
            return self

        def crop(self, tmin, tmax):
            return self

        def apply_baseline(self, baseline):
            return self

        def __getitem__(self, key):
            assert key == "Empty"
            return []

    df = erp_ts_mod.compute_erp_timeseries(
        EmptyConditionEpochs(),
        subject="001",
        channels=["Fz"],
        conditions=["Empty"],
        params=erp_ts_mod.ERPTimeSeriesParams(tmin=-0.1, tmax=0.2, baseline=(-0.1, 0.0)),
    )

    assert df.iloc[0]["status"] == "NO_CONDITIONS"


def test_tfr_safe_pick_and_compute_helpers(monkeypatch, synthetic_epochs):
    assert tfr_mod._safe_pick_channels(synthetic_epochs, ["Fz", "Missing"]) == ["Fz"]

    class DummyEpochs:
        def compute_tfr(self, **kwargs):
            freqs = np.array(kwargs["freqs"], dtype=float)
            n_cycles = np.array(kwargs["n_cycles"], dtype=float)
            assert np.allclose(n_cycles, freqs / 10.0)
            return "power", "itc"

    class DummyEvoked:
        def compute_tfr(self, **kwargs):
            freqs = np.array(kwargs["freqs"], dtype=float)
            n_cycles = np.array(kwargs["n_cycles"], dtype=float)
            assert np.allclose(n_cycles, freqs / 10.0)
            return "evoked_tfr"

    freqs = np.array([2.0, 4.0])
    params = tfr_mod.TFRParams(n_cycles_div=10.0)

    assert tfr_mod._compute_tfr_epochs(DummyEpochs(), freqs, params) == ("power", "itc")
    assert tfr_mod._compute_tfr_evoked(DummyEvoked(), freqs, params) == "evoked_tfr"


def test_compute_tfr_metrics_handles_empty_and_missing_conditions(synthetic_epochs):
    empty_df = tfr_mod.compute_tfr_metrics(
        synthetic_epochs[:0],
        subject="001",
        channels=["Fz"],
        tmin=-0.1,
        tmax=0.1,
    )
    assert empty_df.iloc[0]["status"] == "EMPTY_EPOCHS_OBJECT"

    missing_df = tfr_mod.compute_tfr_metrics(
        synthetic_epochs,
        subject="001",
        channels=["Fz"],
        tmin=-0.1,
        tmax=0.1,
        conditions=["Missing"],
    )
    assert missing_df.iloc[0]["status"] == "MISSING_CONDITION"


def test_compute_tfr_metrics_handles_empty_present_conditions_and_bad_channels():
    class EmptyConditionEpochs:
        ch_names = ["Fz"]
        event_id = {"Empty": 1}

        def __len__(self):
            return 1

        def copy(self):
            return self

        def crop(self, tmin, tmax):
            return self

        def pick(self, chs):
            self.ch_names = list(chs)
            return self

        def __getitem__(self, key):
            assert key == "Empty"
            return []

    empty_df = tfr_mod.compute_tfr_metrics(
        EmptyConditionEpochs(),
        subject="001",
        channels=["Fz"],
        tmin=-0.1,
        tmax=0.1,
        conditions=["Empty"],
    )
    assert empty_df.iloc[0]["status"] == "EMPTY"

    with pytest.raises(ValueError, match="None of the requested channels"):
        tfr_mod.compute_tfr_metrics(
            EmptyConditionEpochs(),
            subject="001",
            channels=["Missing"],
            tmin=-0.1,
            tmax=0.1,
            conditions=["Empty"],
        )


def test_compute_tfr_metrics_returns_flattened_rows(monkeypatch, synthetic_epochs):
    times = np.array([-0.05, 0.0, 0.05])
    freqs = np.array([2.0, 4.0])
    power_total = FakeTFR(
        data=[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
        times=times,
        freqs=freqs,
        ch_names=["Fz"],
    )
    itc = FakeTFR(
        data=[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
        times=times,
        freqs=freqs,
        ch_names=["Fz"],
    )
    evoked_power = FakeTFR(
        data=[[[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]],
        times=times,
        freqs=freqs,
        ch_names=["Fz"],
    )

    monkeypatch.setattr(tfr_mod, "_compute_tfr_epochs", lambda epochs, freqs_in, params: (power_total, itc))
    monkeypatch.setattr(tfr_mod, "_compute_tfr_evoked", lambda evoked, freqs_in, params: evoked_power)

    df = tfr_mod.compute_tfr_metrics(
        synthetic_epochs,
        subject="001",
        channels=["Fz"],
        tmin=-0.05,
        tmax=0.05,
        params=tfr_mod.TFRParams(fmin=2.0, fmax=4.0, fstep=2.0),
        time_decim=2,
    )

    assert set(df["condition"]) == {"Standard", "Deviant"}
    assert len(df) == 8
    assert set(df["status"]) == {"OK"}
    first = df.iloc[0]
    assert first["total_power"] == 1.0
    assert first["evoked_power"] == 0.5
    assert first["induced_power"] == 0.5
    assert first["itc"] == 0.1
    assert first["n_epochs"] == 2


def test_compute_tfr_metrics_rejects_axis_mismatch(monkeypatch, synthetic_epochs):
    times = np.array([-0.05, 0.0])
    freqs = np.array([2.0])
    power_total = FakeTFR(data=[[[1.0, 2.0]]], times=times, freqs=freqs, ch_names=["Fz"])
    itc = FakeTFR(data=[[[0.1, 0.2]]], times=times, freqs=freqs, ch_names=["Fz"])
    evoked_power = FakeTFR(data=[[[0.5, 1.0]]], times=np.array([0.0, 0.1]), freqs=freqs, ch_names=["Fz"])

    monkeypatch.setattr(tfr_mod, "_compute_tfr_epochs", lambda epochs, freqs_in, params: (power_total, itc))
    monkeypatch.setattr(tfr_mod, "_compute_tfr_evoked", lambda evoked, freqs_in, params: evoked_power)

    with pytest.raises(RuntimeError, match="axes do not match"):
        tfr_mod.compute_tfr_metrics(
            synthetic_epochs,
            subject="001",
            channels=["Fz"],
            tmin=-0.05,
            tmax=0.05,
            params=tfr_mod.TFRParams(fmin=2.0, fmax=2.0, fstep=1.0),
        )
