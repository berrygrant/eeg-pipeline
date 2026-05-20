import sys
import types
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


def _parser_args(tmp_path: Path):
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--config", "config.yaml"])
    defaults = cli.build_defaults(parser)

    args.raw_dir = tmp_path / "raw"
    args.subject_csv_dir = tmp_path / "subject_csv"
    args.out_dir = tmp_path / "out"
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.subject_csv_dir.mkdir(parents=True, exist_ok=True)

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
    args.on_missing_subject_csv = "skip"
    args.on_missing_vmrk = "skip"
    args.on_bv_link_mismatch = "skip"
    return args, defaults


def _touch_subject_csv(subject_csv_dir: Path, subj_stem: str):
    subj_num = cli.subject_number_from_stem(subj_stem)
    path = subject_csv_dir / f"subject-{subj_num}.csv"
    path.write_text("EventCode\n1\n2\n1\n2\n", encoding="utf-8")


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
    monkeypatch.setattr(cli, "read_eventcodes_from_subject_csv", lambda path: np.array([1, 2, 1, 2], dtype=int))
    monkeypatch.setattr(cli, "filter_codes", lambda codes, keep: np.asarray(codes, dtype=int))
    monkeypatch.setattr(
        cli,
        "align_marker_positions_to_codes",
        lambda **kwargs: (
            np.array([0, 100, 200, 300], dtype=int),
            {"markers_original": 100, "markers_dropped_by_gap": 2, "markers_dropped_by_auto": 55},
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_events_from_positions_and_codes",
        lambda markers, codes: np.c_[markers, np.zeros(len(codes), dtype=int), np.asarray(codes, dtype=int)],
    )
    monkeypatch.setattr(
        cli,
        "select_and_recode_stddev",
        lambda events, std, dev: (events.copy(), {"Standard": 1, "Deviant": 2}),
    )
    monkeypatch.setattr(
        cli,
        "make_epochs",
        lambda raw, events, event_id, ep: FakeEpochs(n_epochs=n_epochs, event_codes=events[:, 2]),
    )
    monkeypatch.setattr(
        cli,
        "derive_metadata_v1",
        lambda codes, token_map=None: pd.DataFrame({"code": list(codes), "token1": token_map.get("token1", "")}),
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
        lambda rows, path: rows_holder["rows"].extend(rows),
    )

    return rows_holder


def test_run_full_pipeline_happy_path_covers_metrics_ica_and_grand_averages(monkeypatch, tmp_path: Path):
    args, defaults = _parser_args(tmp_path)
    args.ica = "auto"
    args.blink_auto_percentile = 95.0
    args.volt_auto_percentile = 95.0
    args.volt_method = "combined"
    args.volt_step_uv_per_ms = 10.0
    args.save_ica = 1
    args.max_reject_rate = None

    raw_file = args.raw_dir / "S001.vhdr"
    raw_file.write_text("dummy", encoding="utf-8")
    raw_file.with_suffix(".vmrk").write_text("dummy", encoding="utf-8")
    _touch_subject_csv(args.subject_csv_dir, "S001")

    state = _patch_success_dependencies(monkeypatch)

    cli.run_full_pipeline(args, defaults=defaults, cfg={})

    metrics_dir = Path(args.out_dir) / "05_metrics"
    assert (Path(args.out_dir) / "00_ica" / "S001-ica.fif").exists()
    assert (Path(args.out_dir) / "01_clean_raw" / "S001-raw.fif").exists()
    assert (Path(args.out_dir) / "02_epochs" / "S001-epo.fif").exists()
    assert (Path(args.out_dir) / "03_evokeds" / "S001_Standard-ave.fif").exists()
    assert (Path(args.out_dir) / "04_grand_averages" / "grand_average_Standard-ave.fif").exists()
    assert (metrics_dir / "S001_erp_metrics.csv").exists()
    assert (metrics_dir / "S001_tfr_metrics.csv").exists()
    assert (metrics_dir / "erp_metrics_all.csv").exists()
    assert (metrics_dir / "tfr_metrics_all.csv").exists()
    assert (metrics_dir / "erp_timeseries_all.parquet").exists()
    assert (metrics_dir / "erp_timeseries" / "S001_erp_timeseries.parquet").exists()
    assert len(state["rows"]) == 1
    assert state["rows"][0]["status"] == "OK"
    assert state["rows"][0]["review_flag"] is True
    assert state["rows"][0]["ica_ran"] is True
    assert state["rows"][0]["ica_applied"] is True


def test_run_full_pipeline_skips_missing_vmrk_and_writes_qc_only(monkeypatch, tmp_path: Path):
    args, defaults = _parser_args(tmp_path)
    args.on_missing_vmrk = "skip"

    raw_file = args.raw_dir / "S002.vhdr"
    raw_file.write_text("dummy", encoding="utf-8")

    captured = {"rows": []}
    monkeypatch.setattr(cli, "write_qc_summary", lambda rows, path: captured["rows"].extend(rows))

    cli.run_full_pipeline(args, defaults=defaults, cfg={})

    assert len(captured["rows"]) == 1
    assert captured["rows"][0]["status"] == "SKIP_MISSING_VMRK"


def test_run_full_pipeline_skips_bv_link_mismatch(monkeypatch, tmp_path: Path):
    args, defaults = _parser_args(tmp_path)
    args.on_bv_link_mismatch = "skip"

    raw_file = args.raw_dir / "S003.vhdr"
    raw_file.write_text("dummy", encoding="utf-8")
    raw_file.with_suffix(".vmrk").write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(cli, "brainvision_links_ok", lambda path: (False, "Marker/Data mismatch"))
    captured = {"rows": []}
    monkeypatch.setattr(cli, "write_qc_summary", lambda rows, path: captured["rows"].extend(rows))

    cli.run_full_pipeline(args, defaults=defaults, cfg={})

    assert len(captured["rows"]) == 1
    assert captured["rows"][0]["status"] == "SKIP_BV_LINK_MISMATCH"


def test_run_full_pipeline_skips_when_subject_csv_missing(monkeypatch, tmp_path: Path):
    args, defaults = _parser_args(tmp_path)
    args.ica = "off"

    raw_file = args.raw_dir / "S004.set"
    raw_file.write_text("dummy", encoding="utf-8")

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
    monkeypatch.setattr(cli, "write_qc_summary", lambda rows, path: captured["rows"].extend(rows))

    cli.run_full_pipeline(args, defaults=defaults, cfg={})

    assert len(captured["rows"]) == 1
    assert captured["rows"][0]["status"] == "SKIP_MISSING_SUBJECT_CSV"
    assert captured["rows"][0]["trigger_burst_flag"] is True


def test_run_full_pipeline_skips_on_alignment_failure(monkeypatch, tmp_path: Path):
    args, defaults = _parser_args(tmp_path)
    args.ica = "off"

    raw_file = args.raw_dir / "S005.set"
    raw_file.write_text("dummy", encoding="utf-8")
    _touch_subject_csv(args.subject_csv_dir, "S005")

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
            "burst_flag": False,
            "n_short_iti": 0,
            "min_iti_s": 0.4,
            "burst_max_in_window": 1,
            "burst_n_windows_ge_thresh": 0,
            "burst_params": "test",
        },
    )
    monkeypatch.setattr(cli, "read_eventcodes_from_subject_csv", lambda path: np.array([1, 2], dtype=int))
    monkeypatch.setattr(cli, "filter_codes", lambda codes, keep: np.asarray(codes, dtype=int))
    monkeypatch.setattr(
        cli,
        "align_marker_positions_to_codes",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("alignment exploded")),
    )
    captured = {"rows": []}
    monkeypatch.setattr(cli, "write_qc_summary", lambda rows, path: captured["rows"].extend(rows))

    cli.run_full_pipeline(args, defaults=defaults, cfg={})

    assert len(captured["rows"]) == 1
    assert captured["rows"][0]["status"] == "SKIP_ALIGNMENT_FAILED"


def test_run_full_pipeline_skips_when_no_stddev_events(monkeypatch, tmp_path: Path):
    args, defaults = _parser_args(tmp_path)
    args.ica = "off"

    raw_file = args.raw_dir / "S006.set"
    raw_file.write_text("dummy", encoding="utf-8")
    _touch_subject_csv(args.subject_csv_dir, "S006")

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
            "burst_flag": False,
            "n_short_iti": 0,
            "min_iti_s": 0.4,
            "burst_max_in_window": 1,
            "burst_n_windows_ge_thresh": 0,
            "burst_params": "test",
        },
    )
    monkeypatch.setattr(cli, "read_eventcodes_from_subject_csv", lambda path: np.array([1, 2], dtype=int))
    monkeypatch.setattr(cli, "filter_codes", lambda codes, keep: np.asarray(codes, dtype=int))
    monkeypatch.setattr(
        cli,
        "align_marker_positions_to_codes",
        lambda **kwargs: (
            np.array([0, 100], dtype=int),
            {"markers_original": 2, "markers_dropped_by_gap": 0, "markers_dropped_by_auto": 0},
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_events_from_positions_and_codes",
        lambda markers, codes: np.c_[markers, np.zeros(len(codes), dtype=int), np.asarray(codes, dtype=int)],
    )
    monkeypatch.setattr(
        cli,
        "select_and_recode_stddev",
        lambda events, std, dev: (np.empty((0, 3), dtype=int), {"Standard": 1, "Deviant": 2}),
    )
    captured = {"rows": []}
    monkeypatch.setattr(cli, "write_qc_summary", lambda rows, path: captured["rows"].extend(rows))

    cli.run_full_pipeline(args, defaults=defaults, cfg={})

    assert len(captured["rows"]) == 1
    assert captured["rows"][0]["status"] == "SKIP_NO_STDDEV_EVENTS"


def test_run_full_pipeline_skips_when_all_epochs_are_dropped(monkeypatch, tmp_path: Path):
    args, defaults = _parser_args(tmp_path)
    args.ica = "off"
    args.volt_method = "simple"
    args.volt_step_uv_per_ms = None
    args.metrics_erp_timeseries = False

    raw_file = args.raw_dir / "S007.set"
    raw_file.write_text("dummy", encoding="utf-8")
    _touch_subject_csv(args.subject_csv_dir, "S007")

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
        lambda raw: np.array([[0, 0, 1], [100, 0, 2], [200, 0, 1], [300, 0, 2]], dtype=int),
    )
    monkeypatch.setattr(
        cli,
        "detect_trigger_bursts",
        lambda **kwargs: {
            "burst_flag": False,
            "n_short_iti": 0,
            "min_iti_s": 0.4,
            "burst_max_in_window": 1,
            "burst_n_windows_ge_thresh": 0,
            "burst_params": "test",
        },
    )
    monkeypatch.setattr(cli, "read_eventcodes_from_subject_csv", lambda path: np.array([1, 2, 1, 2], dtype=int))
    monkeypatch.setattr(cli, "filter_codes", lambda codes, keep: np.asarray(codes, dtype=int))
    monkeypatch.setattr(
        cli,
        "align_marker_positions_to_codes",
        lambda **kwargs: (
            np.array([0, 100, 200, 300], dtype=int),
            {"markers_original": 4, "markers_dropped_by_gap": 0, "markers_dropped_by_auto": 0},
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_events_from_positions_and_codes",
        lambda markers, codes: np.c_[markers, np.zeros(len(codes), dtype=int), np.asarray(codes, dtype=int)],
    )
    monkeypatch.setattr(
        cli,
        "select_and_recode_stddev",
        lambda events, std, dev: (events.copy(), {"Standard": 1, "Deviant": 2}),
    )
    monkeypatch.setattr(
        cli,
        "make_epochs",
        lambda raw, events, event_id, ep: FakeEpochs(n_epochs=4, event_codes=events[:, 2]),
    )
    monkeypatch.setattr(
        cli,
        "derive_metadata_v1",
        lambda codes, token_map=None: pd.DataFrame({"code": list(codes), "token1": "A"}),
    )
    monkeypatch.setattr(
        cli.mne,
        "pick_types",
        lambda info, eeg=False, eog=False: [2] if eog else ([0, 1] if eeg else []),
    )
    monkeypatch.setattr(cli, "moving_window_ptp_mask", lambda *args, **kwargs: np.ones(4, dtype=bool))
    monkeypatch.setattr(cli, "simple_voltage_threshold_mask", lambda *args, **kwargs: np.zeros(4, dtype=bool))
    captured = {"rows": []}
    monkeypatch.setattr(cli, "write_qc_summary", lambda rows, path: captured["rows"].extend(rows))

    cli.run_full_pipeline(args, defaults=defaults, cfg={})

    assert len(captured["rows"]) == 1
    assert captured["rows"][0]["status"] == "SKIP_EMPTY_EPOCHS"


def test_run_full_pipeline_skips_when_condition_map_has_no_matching_events(monkeypatch, tmp_path: Path):
    args, defaults = _parser_args(tmp_path)
    args.ica = "off"
    args.condition_map = {"Oddball": [7]}

    raw_file = args.raw_dir / "S008.set"
    raw_file.write_text("dummy", encoding="utf-8")
    _touch_subject_csv(args.subject_csv_dir, "S008")

    monkeypatch.setattr(cli, "parse_token_map", lambda token_map: {"token1": "A", "token2": "B"})
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
            "burst_flag": False,
            "n_short_iti": 0,
            "min_iti_s": 0.4,
            "burst_max_in_window": 1,
            "burst_n_windows_ge_thresh": 0,
            "burst_params": "test",
        },
    )
    monkeypatch.setattr(cli, "read_eventcodes_from_subject_csv", lambda path: np.array([1, 2], dtype=int))
    monkeypatch.setattr(cli, "filter_codes", lambda codes, keep: np.asarray(codes, dtype=int))
    monkeypatch.setattr(
        cli,
        "align_marker_positions_to_codes",
        lambda **kwargs: (
            np.array([0, 100], dtype=int),
            {"markers_original": 2, "markers_dropped_by_gap": 0, "markers_dropped_by_auto": 0},
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_events_from_positions_and_codes",
        lambda markers, codes: np.c_[markers, np.zeros(len(codes), dtype=int), np.asarray(codes, dtype=int)],
    )
    monkeypatch.setattr(
        cli,
        "select_and_recode_stddev",
        lambda events, std, dev: (events.copy(), {"Standard": 1, "Deviant": 2}),
    )
    monkeypatch.setattr(
        cli,
        "make_epochs",
        lambda raw, events, event_id, ep: FakeEpochs(n_epochs=2, event_codes=events[:, 2]),
    )
    monkeypatch.setattr(
        cli,
        "select_and_filter_conditions",
        lambda events, condition_map: (np.empty((0, 3), dtype=int), {"Oddball": 7}, [7]),
    )
    monkeypatch.setattr(
        cli,
        "derive_metadata_from_condition_map",
        lambda codes, condition_map: pd.DataFrame({"code": list(codes), "condition": ["Oddball"] * len(codes)}),
    )
    captured = {"rows": []}
    monkeypatch.setattr(cli, "write_qc_summary", lambda rows, path: captured["rows"].extend(rows))

    cli.run_full_pipeline(args, defaults=defaults, cfg={})

    assert len(captured["rows"]) == 1
    assert captured["rows"][0]["status"] == "SKIP_NO_CONDITION_EVENTS"


def test_run_full_pipeline_skips_when_reject_rate_exceeds_limit(monkeypatch, tmp_path: Path):
    args, defaults = _parser_args(tmp_path)
    args.ica = "off"
    args.max_reject_rate = 0.5
    args.volt_method = "simple"
    args.volt_step_uv_per_ms = None
    args.blink_proxy_chs = []

    raw_file = args.raw_dir / "S009.set"
    raw_file.write_text("dummy", encoding="utf-8")
    _touch_subject_csv(args.subject_csv_dir, "S009")

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
        lambda raw: np.array(
            [[0, 0, 1], [100, 0, 2], [200, 0, 1], [300, 0, 2], [400, 0, 1], [500, 0, 2]],
            dtype=int,
        ),
    )
    monkeypatch.setattr(
        cli,
        "detect_trigger_bursts",
        lambda **kwargs: {
            "burst_flag": False,
            "n_short_iti": 0,
            "min_iti_s": 0.4,
            "burst_max_in_window": 1,
            "burst_n_windows_ge_thresh": 0,
            "burst_params": "test",
        },
    )
    monkeypatch.setattr(cli, "read_eventcodes_from_subject_csv", lambda path: np.array([1, 2, 1, 2, 1, 2], dtype=int))
    monkeypatch.setattr(cli, "filter_codes", lambda codes, keep: np.asarray(codes, dtype=int))
    monkeypatch.setattr(
        cli,
        "align_marker_positions_to_codes",
        lambda **kwargs: (
            np.array([0, 100, 200, 300, 400, 500], dtype=int),
            {"markers_original": 6, "markers_dropped_by_gap": 0, "markers_dropped_by_auto": 0},
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_events_from_positions_and_codes",
        lambda markers, codes: np.c_[markers, np.zeros(len(codes), dtype=int), np.asarray(codes, dtype=int)],
    )
    monkeypatch.setattr(
        cli,
        "select_and_recode_stddev",
        lambda events, std, dev: (events.copy(), {"Standard": 1, "Deviant": 2}),
    )
    monkeypatch.setattr(
        cli,
        "make_epochs",
        lambda raw, events, event_id, ep: FakeEpochs(n_epochs=6, event_codes=events[:, 2]),
    )
    monkeypatch.setattr(
        cli,
        "derive_metadata_v1",
        lambda codes, token_map=None: pd.DataFrame({"code": list(codes), "token1": "A"}),
    )
    monkeypatch.setattr(
        cli.mne,
        "pick_types",
        lambda info, eeg=False, eog=False: [] if eog else ([0, 1] if eeg else []),
    )
    monkeypatch.setattr(cli, "simple_voltage_threshold_mask", lambda *args, **kwargs: np.array([1, 1, 1, 1, 0, 0], dtype=bool))
    captured = {"rows": []}
    monkeypatch.setattr(cli, "write_qc_summary", lambda rows, path: captured["rows"].extend(rows))

    cli.run_full_pipeline(args, defaults=defaults, cfg={})

    assert len(captured["rows"]) == 1
    assert captured["rows"][0]["status"] == "SKIP_REJECT_RATE"


def test_run_metrics_only_logs_warnings_when_metric_steps_fail(monkeypatch, tmp_path: Path, capsys):
    epochs_dir = tmp_path / "02_epochs"
    epochs_dir.mkdir(parents=True)
    (epochs_dir / "sub-001-epo.fif").touch()

    args = Namespace(
        out_dir=str(tmp_path),
        metrics_erp_enabled=True,
        metrics_tfr_enabled=True,
        metrics_channels=None,
        metrics_conditions=None,
        condition_map=None,
        compute_mmn=1,
        compute_p300=0,
        difference_label=None,
        metrics_erp_timeseries=True,
        tmin=-0.1,
        tmax=0.2,
        baseline=(-0.1, 0.0),
        tfr_fmin=1.0,
        tfr_fmax=4.0,
        tfr_fstep=1.0,
        tfr_method="multitaper",
        tfr_n_cycles_div=10.0,
        tfr_decim=1,
        tfr_baseline=(-0.1, 0.0),
        tfr_baseline_mode="logratio",
        tfr_tmin=-0.1,
        tfr_tmax=0.2,
        tfr_time_decim=1,
        erp_window=None,
    )

    fake_epochs = FakeEpochs(n_epochs=4)
    monkeypatch.setattr(cli, "_build_erp_windows", lambda args_obj: [cli.ERP_WINDOWS["MMN"]])
    monkeypatch.setattr(cli, "load_epochs", lambda path: SimpleNamespace(epochs=fake_epochs))
    monkeypatch.setattr(cli.mne, "pick_types", lambda info, eeg=True, eog=False: [0, 1])
    monkeypatch.setattr(cli, "compute_erp_metrics", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("erp failed")))
    monkeypatch.setattr(cli, "compute_erp_timeseries", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ts failed")))
    monkeypatch.setattr(cli, "compute_tfr_metrics", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tfr failed")))

    cli.run_metrics_only(args)

    out = capsys.readouterr().out
    assert "ERP metrics failed for sub-001" in out
    assert "ERP timeseries failed for sub-001" in out
    assert "TFR metrics failed for sub-001" in out


def test_run_plot_figures_raises_when_metrics_are_missing_with_stubbed_module(monkeypatch, tmp_path: Path):
    fake_module = types.ModuleType("eeg_pipeline.viz.paper_figures")
    fake_module.main = lambda argv: None
    monkeypatch.setitem(sys.modules, "eeg_pipeline.viz.paper_figures", fake_module)

    args = Namespace(
        out_dir=str(tmp_path),
        figures_out_dir=None,
        figure_time_window=None,
        erp_window=None,
        tmin=-0.2,
        tmax=0.6,
        figure_freq_band=None,
        tfr_fmin=1.0,
        tfr_fmax=30.0,
        figure_diff_heatmap=False,
        figure_channels=None,
    )

    with pytest.raises(FileNotFoundError, match="No metrics found for plotting"):
        cli.run_plot_figures(args)


def test_run_plot_figures_builds_expected_argv_with_stubbed_module(monkeypatch, tmp_path: Path):
    metrics_dir = tmp_path / "05_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "erp_timeseries_all.parquet").write_text("x", encoding="utf-8")
    (metrics_dir / "tfr_metrics_all.csv").write_text("x", encoding="utf-8")

    called = {}
    fake_module = types.ModuleType("eeg_pipeline.viz.paper_figures")
    fake_module.main = lambda argv: called.setdefault("argv", argv)
    monkeypatch.setitem(sys.modules, "eeg_pipeline.viz.paper_figures", fake_module)

    args = Namespace(
        out_dir=str(tmp_path),
        figures_out_dir=str(tmp_path / "figs"),
        figure_time_window=None,
        erp_window=[("MMN", "0.1", "0.2")],
        tmin=-0.2,
        tmax=0.6,
        figure_freq_band=(4.0, 8.0),
        tfr_fmin=1.0,
        tfr_fmax=30.0,
        figure_diff_heatmap=True,
        figure_channels=["Fz", "Cz"],
    )

    cli.run_plot_figures(args)

    argv = called["argv"]
    assert "--out_dir" in argv
    assert "--erp_parquet" in argv
    assert "--tfr_file" in argv
    assert "--freq_band" in argv
    assert "--diff_heatmap" in argv
    assert "--channels" in argv
