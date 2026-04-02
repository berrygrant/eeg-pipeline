from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eeg_pipeline.cli as cli


class _LenOnly:
    def __init__(self, n: int):
        self._n = int(n)

    def __len__(self):
        return self._n


class FakeRaw:
    def __init__(self, sfreq: float = 100.0):
        self.info = {"sfreq": float(sfreq)}

    def save(self, path, overwrite=True):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("raw", encoding="utf-8")


class FakeICA:
    def save(self, path, overwrite=True):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("ica", encoding="utf-8")


class FakeEvoked:
    def __init__(self, nave: int = 2):
        self.nave = nave

    def save(self, path, overwrite=True):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("evoked", encoding="utf-8")


class FakeEpochs:
    def __init__(self, n_epochs=4, event_codes=None, ch_names=None):
        if ch_names is None:
            ch_names = ["Fz", "Cz", "EOG"]
        self.ch_names = list(ch_names)
        self.info = {"sfreq": 100.0}
        self._data = np.zeros((int(n_epochs), len(self.ch_names), 20), dtype=float)
        if event_codes is None:
            event_codes = [1 if (i % 2 == 0) else 2 for i in range(int(n_epochs))]
        event_codes = np.asarray(event_codes, dtype=int)
        self.events = np.c_[np.arange(len(event_codes), dtype=int), np.zeros(len(event_codes), dtype=int), event_codes]
        self.event_id = {"Standard": 1, "Deviant": 2}
        self.selection = np.arange(len(event_codes), dtype=int)
        self.metadata = None

    def __len__(self):
        return int(self._data.shape[0])

    def __getitem__(self, key):
        if isinstance(key, str):
            code = self.event_id[key]
            return _LenOnly(int(np.sum(self.events[:, 2] == code)))
        raise TypeError("FakeEpochs only supports string indexing.")

    def copy(self):
        out = FakeEpochs(n_epochs=1)
        out.ch_names = list(self.ch_names)
        out.info = dict(self.info)
        out._data = self._data.copy()
        out.events = self.events.copy()
        out.event_id = dict(self.event_id)
        out.selection = self.selection.copy()
        out.metadata = None if self.metadata is None else self.metadata.copy()
        return out

    def crop(self, tmin=None, tmax=None):
        return self

    def get_data(self, picks=None):
        if picks is None:
            return self._data
        return self._data[:, list(picks), :]

    def drop(self, bad_idx, reason=None):
        keep = np.ones(len(self), dtype=bool)
        if bad_idx:
            keep[np.asarray(bad_idx, dtype=int)] = False
        self._data = self._data[keep]
        self.events = self.events[keep]
        self.selection = self.selection[keep]
        if isinstance(self.metadata, pd.DataFrame):
            self.metadata = self.metadata.iloc[keep].reset_index(drop=True)

    def save(self, path, overwrite=True):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("epo", encoding="utf-8")


def _make_bids_fixture(tmp_path: Path, *, subject: str = "01", with_events: bool = True, ext: str = ".vhdr") -> tuple[Path, Path]:
    bids_root = tmp_path / "bids"
    derivatives_root = tmp_path / "derivatives"
    bids_root.mkdir(parents=True)
    (bids_root / "dataset_description.json").write_text('{"Name":"Fixture","BIDSVersion":"1.11.1"}', encoding="utf-8")
    eeg_dir = bids_root / f"sub-{subject}" / "eeg"
    eeg_dir.mkdir(parents=True)
    raw_path = eeg_dir / f"sub-{subject}_task-oddball_run-01_eeg{ext}"
    raw_path.write_text("MarkerFile=sub.vmrk\nDataFile=sub.eeg\n", encoding="utf-8")
    if ext == ".vhdr":
        raw_path.with_suffix(".vmrk").write_text("dummy", encoding="utf-8")
        raw_path.with_suffix(".eeg").write_text("dummy", encoding="utf-8")
    if with_events:
        (eeg_dir / f"sub-{subject}_task-oddball_run-01_events.tsv").write_text(
            "onset\tduration\tsample\ttrial_type\tvalue\n0.0\t0.1\t0\tStandard\t1\n1.0\t0.1\t100\tDeviant\t2\n2.0\t0.1\t200\tStandard\t1\n3.0\t0.1\t300\tDeviant\t2\n",
            encoding="utf-8",
        )
        (eeg_dir / f"sub-{subject}_task-oddball_run-01_events.json").write_text(
            '{"trial_type":{"Description":"Condition"}}',
            encoding="utf-8",
        )
    return bids_root, derivatives_root


def _parser_args(tmp_path: Path):
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--config", "config.yaml"])
    defaults = cli.build_defaults(parser)

    bids_root, derivatives_root = _make_bids_fixture(tmp_path)
    args.bids_root = bids_root
    args.derivatives_root = derivatives_root
    args.standard_codes = [1]
    args.deviant_codes = [2]
    args.behavioral_keep_codes = [1, 2]
    args.metrics = 1
    args.metrics_erp_enabled = True
    args.metrics_tfr_enabled = True
    args.metrics_erp_timeseries = True
    args.metrics_channels = None
    args.token_map = ["token1=A", "token2=B"]
    args.blink_proxy_chs = ["Fz"]
    args.eog_chs = ["EOG"]
    args.on_missing_vmrk = "skip"
    args.on_bv_link_mismatch = "skip"
    args.behavior_csv_fallback_dir = None
    return args, defaults


def _patch_success_dependencies(monkeypatch, *, n_epochs=4, burst_flag=True):
    rows_holder = {"rows": []}

    monkeypatch.setattr(cli, "parse_token_map", lambda token_map: {"token1": "A", "token2": "B"})
    monkeypatch.setattr(cli, "brainvision_links_ok", lambda path: (True, ""))
    monkeypatch.setattr(cli, "read_raw_preprocess", lambda **kwargs: FakeRaw())
    monkeypatch.setattr(
        cli,
        "compute_ica_diagnostics",
        lambda raw, **kwargs: {
            "eog_corr_max": 0.5,
            "eog_corr_mean": 0.2,
            "blink_rate_per_min": 20.0,
            "blink_proxy_rate_per_min": 0.0,
            "blink_source": "EOG",
        },
    )
    monkeypatch.setattr(cli, "fit_ica", lambda raw, params: (FakeICA(), {"status": "fit_ok"}))
    monkeypatch.setattr(cli, "find_ica_excludes", lambda *args, **kwargs: ([0], {"excluded": 1}))
    monkeypatch.setattr(cli, "apply_ica", lambda raw, ica_obj, exclude: raw)
    monkeypatch.setattr(
        cli,
        "events_from_annotations_positions",
        lambda raw: np.array(
            [[0, 0, 1], [100, 0, 2], [200, 0, 1], [300, 0, 2]],
            dtype=int,
        ),
    )
    monkeypatch.setattr(
        cli,
        "detect_trigger_bursts",
        lambda **kwargs: {
            "burst_flag": bool(burst_flag),
            "n_short_iti": 3 if burst_flag else 0,
            "min_iti_s": 0.01 if burst_flag else 0.2,
            "burst_max_in_window": 6 if burst_flag else 2,
            "burst_n_windows_ge_thresh": 1 if burst_flag else 0,
            "burst_params": "test",
        },
    )
    monkeypatch.setattr(
        cli,
        "make_epochs",
        lambda raw, events, event_id, ep: FakeEpochs(n_epochs=n_epochs, event_codes=events[:, 2]),
    )
    monkeypatch.setattr(
        cli.mne,
        "pick_types",
        lambda info, eeg=False, eog=False: [2] if eog else ([0, 1] if eeg else []),
    )
    monkeypatch.setattr(cli.mne, "pick_channels", lambda ch_names, include: [0] if include else [])
    monkeypatch.setattr(cli, "moving_window_ptp_max", lambda *args, **kwargs: np.array([1.0, 2.0, 3.0, 4.0]))

    def _ptp_mask(data, **kwargs):
        if data.shape[1] == 1:
            return np.array([False, True, False, False], dtype=bool)
        return np.array([False, False, True, False], dtype=bool)

    monkeypatch.setattr(cli, "moving_window_ptp_mask", _ptp_mask)
    monkeypatch.setattr(
        cli, "simple_voltage_threshold_mask", lambda *args, **kwargs: np.zeros(n_epochs, dtype=bool)
    )
    monkeypatch.setattr(
        cli, "step_threshold_mask", lambda *args, **kwargs: np.array([False, True, False, False], dtype=bool)
    )
    monkeypatch.setattr(
        cli, "compute_erp_metrics", lambda *args, **kwargs: pd.DataFrame([{"subject": kwargs["subject"], "ok": 1}])
    )
    monkeypatch.setattr(
        cli, "compute_erp_timeseries", lambda *args, **kwargs: pd.DataFrame([{"subject": kwargs["subject"], "ok": 1}])
    )
    monkeypatch.setattr(
        cli, "compute_tfr_metrics", lambda *args, **kwargs: pd.DataFrame([{"subject": kwargs["subject"], "ok": 1}])
    )
    monkeypatch.setattr(cli, "recommend_ica", lambda **kwargs: {"ica_recommended": True, "ica_recommend_reason": "test"})
    monkeypatch.setattr(cli, "compute_evokeds", lambda epochs, conds: {cond: FakeEvoked() for cond in conds})
    monkeypatch.setattr(cli, "grand_averages", lambda evokeds_by_cond: {cond: FakeEvoked() for cond in evokeds_by_cond})
    monkeypatch.setattr(
        cli,
        "write_qc_summary",
        lambda rows, path: rows_holder["rows"].extend(rows) or pd.DataFrame(rows).to_csv(path, sep="\t", index=False),
    )

    return rows_holder


def test_run_full_pipeline_happy_path_writes_bids_derivatives(monkeypatch, tmp_path: Path):
    args, defaults = _parser_args(tmp_path)
    args.ica = "auto"
    args.blink_auto_percentile = 95.0
    args.volt_auto_percentile = 95.0
    args.volt_method = "combined"
    args.volt_step_uv_per_ms = 10.0
    args.save_ica = 1
    args.max_reject_rate = None

    state = _patch_success_dependencies(monkeypatch)

    cli.run_full_pipeline(args, defaults=defaults, cfg={})

    dataset_root = Path(args.derivatives_root) / "eeg-pipeline"
    subject_root = dataset_root / "sub-01" / "eeg"
    dataset_metrics = dataset_root / "eeg"
    assert (dataset_root / "dataset_description.json").exists()
    assert (subject_root / "sub-01_task-oddball_run-01_desc-components_ica.fif").exists()
    assert (subject_root / "sub-01_task-oddball_run-01_desc-preproc_eeg.fif").exists()
    assert (subject_root / "sub-01_task-oddball_run-01_epo.fif").exists()
    assert (subject_root / "sub-01_task-oddball_run-01_desc-standard_ave.fif").exists()
    assert (subject_root / "sub-01_task-oddball_run-01_desc-aligned_events.tsv").exists()
    assert (subject_root / "sub-01_task-oddball_run-01_desc-aligned_events.json").exists()
    assert (subject_root / "sub-01_task-oddball_run-01_desc-erp_metrics.tsv").exists()
    assert (subject_root / "sub-01_task-oddball_run-01_desc-tfr_metrics.tsv").exists()
    assert (subject_root / "sub-01_task-oddball_run-01_desc-erp_timeseries.parquet").exists()
    assert (dataset_metrics / "desc-erp_metrics.tsv").exists()
    assert (dataset_metrics / "desc-tfr_metrics.tsv").exists()
    assert (dataset_metrics / "desc-erp_timeseries.parquet").exists()
    assert (dataset_metrics / "desc-summary_qc.tsv").exists()
    assert (dataset_metrics / "task-oddball_desc-grandaverage-standard_ave.fif").exists()
    assert len(state["rows"]) == 1
    assert state["rows"][0]["status"] == "OK"
    assert state["rows"][0]["behavior_source"] == "bids_events"
    assert state["rows"][0]["review_flag"] is True
    assert state["rows"][0]["ica_ran"] is True
    assert state["rows"][0]["ica_applied"] is True


def test_run_full_pipeline_skips_missing_bids_events(monkeypatch, tmp_path: Path):
    args, defaults = _parser_args(tmp_path)
    args.ica = "off"
    bids_root, derivatives_root = _make_bids_fixture(tmp_path / "missing", with_events=False)
    args.bids_root = bids_root
    args.derivatives_root = derivatives_root

    monkeypatch.setattr(cli, "read_raw_preprocess", lambda **kwargs: FakeRaw())
    monkeypatch.setattr(
        cli,
        "compute_ica_diagnostics",
        lambda raw, **kwargs: {
            "eog_corr_max": 0.1,
            "eog_corr_mean": 0.1,
            "blink_rate_per_min": 1.0,
            "blink_proxy_rate_per_min": 0.0,
            "blink_source": "EOG",
        },
    )
    monkeypatch.setattr(
        cli,
        "events_from_annotations_positions",
        lambda raw: np.array([[0, 0, 1], [100, 0, 2]], dtype=int),
    )
    monkeypatch.setattr(
        cli,
        "detect_trigger_bursts",
        lambda **kwargs: {
            "burst_flag": True,
            "n_short_iti": 1,
            "min_iti_s": 0.01,
            "burst_max_in_window": 5,
            "burst_n_windows_ge_thresh": 1,
            "burst_params": "test",
        },
    )
    captured = {"rows": []}
    monkeypatch.setattr(
        cli,
        "write_qc_summary",
        lambda rows, path: captured["rows"].extend(rows) or pd.DataFrame(rows).to_csv(path, sep="\t", index=False),
    )

    cli.run_full_pipeline(args, defaults=defaults, cfg={})

    assert len(captured["rows"]) == 1
    assert captured["rows"][0]["status"] == "SKIP_MISSING_EVENTS"
    assert captured["rows"][0]["trigger_burst_flag"] is True


def test_run_full_pipeline_skips_missing_vmrk_and_writes_qc_only(monkeypatch, tmp_path: Path):
    args, defaults = _parser_args(tmp_path)
    args.on_missing_vmrk = "skip"
    bids_root = args.bids_root
    raw_file = bids_root / "sub-01" / "eeg" / "sub-01_task-oddball_run-01_eeg.vhdr"
    raw_file.with_suffix(".vmrk").unlink()

    captured = {"rows": []}
    monkeypatch.setattr(
        cli,
        "write_qc_summary",
        lambda rows, path: captured["rows"].extend(rows) or pd.DataFrame(rows).to_csv(path, sep="\t", index=False),
    )

    cli.run_full_pipeline(args, defaults=defaults, cfg={})

    assert len(captured["rows"]) == 1
    assert captured["rows"][0]["status"] == "SKIP_MISSING_VMRK"
