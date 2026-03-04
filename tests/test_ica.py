from types import SimpleNamespace

import numpy as np

import eeg_pipeline.ica as ica_mod


class DummyRaw:
    def __init__(self, ch_names):
        self.info = {"ch_names": list(ch_names)}
        self.filter_calls = []
        self.data_by_pick = {}

    def copy(self):
        clone = DummyRaw(self.info["ch_names"])
        clone.data_by_pick = dict(self.data_by_pick)
        return clone

    def filter(self, **kwargs):
        self.filter_calls.append(kwargs)
        return self

    def get_data(self, picks=None):
        if picks is None:
            raise AssertionError("Expected picks to be provided in this test")
        key = tuple(picks)
        if key in self.data_by_pick:
            return self.data_by_pick[key]
        if len(picks) == 1 and picks[0] in self.data_by_pick:
            return self.data_by_pick[picks[0]]
        raise KeyError(f"Unexpected picks {picks!r}")


def test_safe_pick_channels_returns_existing_indices_only():
    info = {"ch_names": ["Fz", "Cz", "EOG"]}

    picks = ica_mod._safe_pick_channels(info, ["EOG", "Missing", "Fz"])

    assert picks == [2, 0]


def test_fit_ica_returns_none_when_too_few_eeg_channels(monkeypatch):
    raw = DummyRaw(["Fz"])
    monkeypatch.setattr(ica_mod.mne, "pick_types", lambda *args, **kwargs: np.array([0]))

    ica, diag = ica_mod.fit_ica(raw, ica_mod.ICAParams())

    assert ica is None
    assert diag["ica_fit_ok"] is False
    assert diag["ica_fit_n_eeg_chs"] == 1
    assert "Need >=2 EEG channels" in diag["ica_fit_error"]


def test_fit_ica_retries_after_variance_fraction_runtime_error(monkeypatch):
    raw = DummyRaw(["Fz", "Cz", "Pz", "Oz", "P3"])
    monkeypatch.setattr(ica_mod.mne, "pick_types", lambda *args, **kwargs: np.array([0, 1, 2, 3, 4]))

    class FakeICA:
        def __init__(self, method, n_components, random_state, max_iter):
            self.n_components = n_components

        def fit(self, raw_fit, picks, decim, verbose=False):
            if isinstance(self.n_components, float):
                raise RuntimeError("your threshold results in 1 component")
            return self

    monkeypatch.setattr(ica_mod.mne.preprocessing, "ICA", FakeICA)

    fitted, diag = ica_mod.fit_ica(raw, ica_mod.ICAParams(n_components=0.99))

    assert isinstance(fitted, FakeICA)
    assert fitted.n_components == 15
    assert diag["ica_fit_ok"] is True
    assert diag["ica_fit_n_components_used"] == 15
    assert diag["ica_fit_retry"] == "retry_int_n_components=15"


def test_fit_ica_returns_none_on_nonretryable_runtime_error(monkeypatch):
    raw = DummyRaw(["Fz", "Cz"])
    monkeypatch.setattr(ica_mod.mne, "pick_types", lambda *args, **kwargs: np.array([0, 1]))

    class FakeICA:
        def __init__(self, method, n_components, random_state, max_iter):
            pass

        def fit(self, raw_fit, picks, decim, verbose=False):
            raise RuntimeError("unexpected failure")

    monkeypatch.setattr(ica_mod.mne.preprocessing, "ICA", FakeICA)

    fitted, diag = ica_mod.fit_ica(raw, ica_mod.ICAParams())

    assert fitted is None
    assert diag["ica_fit_ok"] is False
    assert diag["ica_fit_retry"] == ""
    assert diag["ica_fit_error"] == "unexpected failure"


def test_find_ica_excludes_uses_eog_candidates_sorted_by_absolute_score():
    raw = DummyRaw(["EOG", "Fz"])

    class FakeICA:
        def find_bads_eog(self, raw_obj, ch_name, threshold="auto", verbose=False):
            return [2, 0, 1], np.array([-0.9, 0.5, 0.2])

    exclude, diag = ica_mod.find_ica_excludes(
        FakeICA(),
        raw,
        eog_chs=["EOG"],
        max_exclude=2,
    )

    assert exclude == [0, 1]
    assert diag["ica_blink_source"] == "eog"
    assert diag["ica_eog_channel_used"] == "EOG"
    assert diag["ica_candidates"] == [0, 1, 2]


def test_find_ica_excludes_falls_back_to_proxy_correlations():
    raw = DummyRaw(["Fz", "Cz"])
    raw.data_by_pick = {
        ("Fz",): np.array([[0.0, 1.0, 2.0]]),
    }

    class FakeSources:
        def get_data(self):
            return np.array(
                [
                    [0.0, 1.0, 2.0],
                    [2.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                ]
            )

    class FakeICA:
        def get_sources(self, raw_obj):
            return FakeSources()

    exclude, diag = ica_mod.find_ica_excludes(
        FakeICA(),
        raw,
        proxy_chs=["Fz"],
        corr_thresh=0.8,
        max_exclude=2,
    )

    assert exclude == [0, 1]
    assert diag["ica_blink_source"] == "proxy"
    assert diag["ica_proxy_channel_used"] == "Fz"
    assert np.allclose(diag["ica_excluded_scores"], [1.0, -1.0])


def test_find_ica_excludes_handles_missing_proxy_channels():
    raw = DummyRaw(["Cz"])

    class FakeICA:
        pass

    exclude, diag = ica_mod.find_ica_excludes(FakeICA(), raw, proxy_chs=["Fp1"])

    assert exclude == []
    assert diag["ica_proxy_channel_used"] == ""
    assert diag["ica_error"] == "no_eog_and_no_proxy_channel_found"


def test_apply_ica_uses_copy_and_sets_integer_excludes():
    original = DummyRaw(["Fz", "Cz"])
    cleaned = DummyRaw(["Fz", "Cz"])
    original.copy = lambda: cleaned

    class FakeICA:
        def __init__(self):
            self.exclude = []
            self.applied = None

        def apply(self, raw_obj, verbose=False):
            self.applied = (raw_obj, verbose)

    fake_ica = FakeICA()

    result = ica_mod.apply_ica(original, fake_ica, exclude=[1, "2"])

    assert result is cleaned
    assert fake_ica.exclude == [1, 2]
    assert fake_ica.applied == (cleaned, False)
