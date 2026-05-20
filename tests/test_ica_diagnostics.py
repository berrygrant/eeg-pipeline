import numpy as np

import eeg_pipeline.ica_diagnostics as ica_diagnostics


def test_count_clusters_and_recommend_ica_cover_positive_and_negative_cases():
    assert ica_diagnostics.count_clusters(np.array([], dtype=bool)) == 0
    assert ica_diagnostics.count_clusters(np.array([0, 1, 1, 0, 1], dtype=bool)) == 2

    rec = ica_diagnostics.recommend_ica(
        epoch_reject_rate=0.30,
        eog_corr_max=0.50,
        blink_rate_per_min=np.nan,
        blink_proxy_rate_per_min=25.0,
    )
    assert rec["ica_recommended"] is True
    assert "blink_proxy>20/min" in rec["ica_recommend_reason"]
    assert "epoch_loss>0.20" in rec["ica_recommend_reason"]
    assert "eog_corr>0.30" in rec["ica_recommend_reason"]

    assert ica_diagnostics.recommend_ica(
        epoch_reject_rate=0.05,
        eog_corr_max=0.10,
        blink_rate_per_min=5.0,
        blink_proxy_rate_per_min=0.0,
    ) == {"ica_recommended": False, "ica_recommend_reason": ""}

    rec = ica_diagnostics.recommend_ica(
        epoch_reject_rate=0.0,
        eog_corr_max=0.0,
        blink_rate_per_min=25.0,
        blink_proxy_rate_per_min=0.0,
    )
    assert "blink_rate>20/min" in rec["ica_recommend_reason"]


def test_compute_ica_diagnostics_uses_true_eog_when_available(monkeypatch, synthetic_raw):
    monkeypatch.setattr(
        ica_diagnostics,
        "find_eog_events",
        lambda raw, ch_name, verbose=False: np.array([[1, 0, 1], [2, 0, 1]]),
    )

    metrics = ica_diagnostics.compute_ica_diagnostics(synthetic_raw)

    assert np.isfinite(metrics["eog_corr_max"])
    assert metrics["blink_rate_per_min"] > 0
    assert metrics["blink_source"] == "eog:EOG"


def test_compute_ica_diagnostics_falls_back_to_proxy_channel(monkeypatch, synthetic_raw):
    raw = synthetic_raw.copy()
    raw.drop_channels(["EOG"])

    monkeypatch.setattr(
        ica_diagnostics,
        "find_eog_events",
        lambda raw, ch_name, verbose=False: np.array([[1, 0, 1]]),
    )

    metrics = ica_diagnostics.compute_ica_diagnostics(raw, blink_proxy_chs=["Fz"])

    assert metrics["blink_proxy_rate_per_min"] > 0
    assert metrics["blink_source"] == "proxy:Fz"


def test_compute_ica_diagnostics_handles_eog_and_proxy_detector_failures(monkeypatch, synthetic_raw):
    monkeypatch.setattr(
        ica_diagnostics,
        "find_eog_events",
        lambda raw, ch_name, verbose=False: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    metrics = ica_diagnostics.compute_ica_diagnostics(synthetic_raw)
    assert metrics["blink_rate_per_min"] == 0.0
    assert metrics["blink_source"] == "eog:EOG"

    raw = synthetic_raw.copy()
    raw.drop_channels(["EOG"])

    proxy_metrics = ica_diagnostics.compute_ica_diagnostics(raw, blink_proxy_chs=["Fz"])
    assert proxy_metrics["blink_proxy_rate_per_min"] == 0.0
    assert proxy_metrics["blink_source"] == "proxy:Fz"

    none_metrics = ica_diagnostics.compute_ica_diagnostics(raw, blink_proxy_chs=["Missing"])
    assert none_metrics["blink_source"] == "none"


def test_compute_ica_diagnostics_handles_zero_duration_without_trying_blink_detection(monkeypatch):
    class RawStub:
        info = {"sfreq": 100.0}
        n_times = 0
        ch_names = []

    monkeypatch.setattr(ica_diagnostics.mne, "pick_types", lambda info, eog=False, eeg=False: [])

    metrics = ica_diagnostics.compute_ica_diagnostics(RawStub())

    assert metrics["blink_source"] == "none"
    assert np.isnan(metrics["blink_rate_per_min"])
    assert np.isnan(metrics["blink_proxy_rate_per_min"])
