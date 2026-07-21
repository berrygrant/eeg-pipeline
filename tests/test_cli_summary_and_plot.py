import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eeg_pipeline.cli as cli
import eeg_pipeline.cli_summary as cli_summary


def _summary_args(bids_root: Path) -> Namespace:
    return Namespace(
        bids_root=str(bids_root),
        behavior_csv_fallback_dir=None,
        montage="standard_1020",
        eog_chs=["EOG"],
        aux_chs=[],
        reref="average",
        l_freq=0.1,
        h_freq=20.0,
        notch=None,
        blink_proxy_chs=["Fp1", "Fp2"],
        blink_threshold_uv=100.0,
        blink_win_ms=150.0,
        blink_step_ms=10.0,
        ica_corr_thresh=0.3,
        ica_auto_blink_rate_per_min=10.0,
        behavioral_keep_codes=[1, 2],
        drop_eeg_markers_by_gap_s=1.5,
        auto_drop_to_count=False,
        token_map=["token1=A", "token2=B"],
        condition_map=None,
    )


def _make_bids_summary_fixture(tmp_path: Path, *, with_events: bool = True) -> tuple[Path, Path]:
    bids_root = tmp_path / "bids"
    bids_root.mkdir(parents=True)
    (bids_root / "dataset_description.json").write_text('{"Name":"Fixture","BIDSVersion":"1.11.1"}', encoding="utf-8")
    eeg_dir = bids_root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    raw_path = eeg_dir / "sub-01_task-oddball_run-01_eeg.vhdr"
    raw_path.write_text("dummy", encoding="utf-8")
    raw_path.with_suffix(".vmrk").write_text("dummy", encoding="utf-8")
    if with_events:
        (eeg_dir / "sub-01_task-oddball_run-01_events.tsv").write_text(
            "onset\tduration\tsample\ttrial_type\tvalue\n0.0\t0.1\t100\tStandard\t1\n1.0\t0.1\t200\tDeviant\t2\n",
            encoding="utf-8",
        )
    return bids_root, raw_path


def _patch_summary_dependencies(monkeypatch, synthetic_raw):
    raw0 = SimpleNamespace(
        annotations=SimpleNamespace(description=["Stimulus/S1", "Stimulus/S1", "Response/R1"])
    )

    monkeypatch.setattr(cli_summary.mne.io, "read_raw_brainvision", lambda *args, **kwargs: raw0)
    monkeypatch.setattr(cli_summary.mne.io, "read_raw_eeglab", lambda *args, **kwargs: raw0)
    monkeypatch.setattr(cli_summary, "read_raw_preprocess", lambda **kwargs: synthetic_raw.copy())
    monkeypatch.setattr(
        cli_summary,
        "compute_ica_diagnostics",
        lambda *args, **kwargs: {
            "eog_corr_max": 0.45,
            "eog_corr_mean": 0.22,
            "blink_rate_per_min": 12.0,
            "blink_proxy_rate_per_min": 0.0,
        },
    )
    monkeypatch.setattr(
        cli_summary,
        "recommend_ica",
        lambda **kwargs: {"ica_recommended": True, "ica_recommend_reason": "blink_rate"},
    )
    monkeypatch.setattr(
        cli_summary,
        "events_from_annotations_positions",
        lambda raw: np.array([[0, 0, 1], [1, 0, 1], [2, 0, 1], [100, 0, 1]], dtype=int),
    )
    monkeypatch.setattr(
        cli_summary,
        "marker_gap_stats",
        lambda markers_pos, sfreq: {
            "dt_min": 0.01,
            "dt_p25": 0.01,
            "dt_p50": 0.01,
            "dt_p75": 0.49,
            "dt_p90": 0.98,
            "dt_p95": 0.98,
            "dt_p99": 0.98,
            "dt_max": 0.98,
        },
    )
    monkeypatch.setattr(cli_summary, "keep_by_gap_heuristic", lambda markers_pos, sfreq, gap_s: [0, 3])
    monkeypatch.setattr(
        cli_summary,
        "parse_vmrk_markers",
        lambda path: pd.DataFrame({"mtype": ["Stimulus", "Response"], "desc": ["S 1", "R 1"]}),
    )
    monkeypatch.setattr(cli_summary, "parse_token_map", lambda token_map: {"token1": "A", "token2": "B"})


def test_summarize_one_file_reports_bids_events(monkeypatch, tmp_path: Path, synthetic_raw, capsys):
    bids_root, raw_path = _make_bids_summary_fixture(tmp_path)
    args = _summary_args(bids_root)
    _patch_summary_dependencies(monkeypatch, synthetic_raw)

    cli.summarize_one_file(args, raw_path)

    out = capsys.readouterr().out
    assert "SUMMARY: sub-01" in out
    assert "BIDS events:" in out
    assert "Using BIDS events.tsv sample column directly" in out
    assert "Markers from .vmrk:" in out
    assert "Token map: {'token1': 'A', 'token2': 'B'}" in out


def test_summarize_one_file_warns_when_bids_events_are_missing(monkeypatch, tmp_path: Path, synthetic_raw, capsys):
    bids_root, raw_path = _make_bids_summary_fixture(tmp_path, with_events=False)
    args = _summary_args(bids_root)
    _patch_summary_dependencies(monkeypatch, synthetic_raw)

    cli.summarize_one_file(args, raw_path)

    out = capsys.readouterr().out
    assert "Missing BIDS events file" in out
    assert "Exiting summary." in out


def test_main_exits_early_for_one_file_summary(monkeypatch):
    called = {}

    monkeypatch.setattr(cli, "apply_erp_core_preset", lambda args, defaults: None)
    monkeypatch.setattr(cli, "apply_config", lambda args, defaults: {"cfg": "ok"})
    monkeypatch.setattr(cli, "summarize_one_file", lambda args, path: called.setdefault("path", path))

    cli.main(["--config", "config.yaml", "--summarize_one_file", "demo.vhdr"])

    assert called["path"] == Path("demo.vhdr")


def test_run_plot_figures_raises_when_metrics_are_missing_with_stubbed_module(monkeypatch, tmp_path: Path):
    fake_module = types.ModuleType("eeg_pipeline.viz.paper_figures")
    fake_module.main = lambda argv: None
    monkeypatch.setitem(sys.modules, "eeg_pipeline.viz.paper_figures", fake_module)

    args = Namespace(
        bids_root=str(tmp_path / "bids"),
        derivatives_root=str(tmp_path / "derivatives"),
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
    dataset_root = tmp_path / "derivatives" / "eeg-pipeline"
    metrics_dir = dataset_root / "eeg"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "desc-erp_timeseries.parquet").write_text("x", encoding="utf-8")
    (metrics_dir / "desc-tfr_metrics.tsv").write_text("x", encoding="utf-8")

    called = {}
    fake_module = types.ModuleType("eeg_pipeline.viz.paper_figures")
    fake_module.main = lambda argv: called.setdefault("argv", argv)
    monkeypatch.setitem(sys.modules, "eeg_pipeline.viz.paper_figures", fake_module)

    args = Namespace(
        bids_root=str(tmp_path / "bids"),
        derivatives_root=str(tmp_path / "derivatives"),
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
