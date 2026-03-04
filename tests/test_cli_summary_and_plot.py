from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eeg_pipeline.cli as cli


def _summary_args(subject_csv_dir: Path) -> Namespace:
    return Namespace(
        subject_csv_dir=str(subject_csv_dir),
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
    )


def _patch_summary_dependencies(monkeypatch, synthetic_raw):
    raw0 = SimpleNamespace(
        annotations=SimpleNamespace(description=["Stimulus/S1", "Stimulus/S1", "Response/R1"])
    )

    monkeypatch.setattr(cli.mne.io, "read_raw_brainvision", lambda *args, **kwargs: raw0)
    monkeypatch.setattr(cli.mne.io, "read_raw_eeglab", lambda *args, **kwargs: raw0)
    monkeypatch.setattr(cli, "read_raw_preprocess", lambda **kwargs: synthetic_raw.copy())
    monkeypatch.setattr(
        cli,
        "compute_ica_diagnostics",
        lambda *args, **kwargs: {
            "eog_corr_max": 0.45,
            "eog_corr_mean": 0.22,
            "blink_rate_per_min": 12.0,
            "blink_proxy_rate_per_min": 0.0,
        },
    )
    monkeypatch.setattr(
        cli,
        "recommend_ica",
        lambda **kwargs: {"ica_recommended": True, "ica_recommend_reason": "blink_rate"},
    )
    monkeypatch.setattr(
        cli,
        "events_from_annotations_positions",
        lambda raw: np.array([[0, 0, 1], [1, 0, 1], [2, 0, 1], [100, 0, 1]], dtype=int),
    )
    monkeypatch.setattr(
        cli,
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
    monkeypatch.setattr(cli, "keep_by_gap_heuristic", lambda markers_pos, sfreq, gap_s: [0, 3])
    monkeypatch.setattr(
        cli,
        "parse_vmrk_markers",
        lambda path: pd.DataFrame({"mtype": ["Stimulus", "Response"], "desc": ["S 1", "R 1"]}),
    )
    monkeypatch.setattr(cli, "read_eventcodes_from_subject_csv", lambda path: np.array([1, 2, 99]))
    monkeypatch.setattr(cli, "filter_codes", lambda codes, keep_codes: np.array([1, 2]))
    monkeypatch.setattr(
        cli,
        "align_marker_positions_to_codes",
        lambda **kwargs: (
            np.array([100, 200], dtype=int),
            {
                "markers_original": 4,
                "markers_dropped_by_gap": 1,
                "markers_dropped_by_auto": 1,
            },
        ),
    )
    monkeypatch.setattr(cli, "parse_token_map", lambda token_map: {"token1": "A", "token2": "B"})
    monkeypatch.setattr(
        cli,
        "derive_metadata_v1",
        lambda codes, token_map=None: pd.DataFrame(
            [{"code": int(code), "token1": token_map["token1"]} for code in codes]
        ),
    )


def test_summarize_one_file_brainvision_success(monkeypatch, tmp_path: Path, synthetic_raw, capsys):
    subject_csv_dir = tmp_path / "subjects"
    subject_csv_dir.mkdir()
    (subject_csv_dir / "subject-203.csv").write_text("placeholder", encoding="utf-8")

    raw_path = tmp_path / "S203.vhdr"
    raw_path.write_text("dummy", encoding="utf-8")
    raw_path.with_suffix(".vmrk").write_text("dummy", encoding="utf-8")

    args = _summary_args(subject_csv_dir)
    _patch_summary_dependencies(monkeypatch, synthetic_raw)

    cli.summarize_one_file(args, raw_path)

    out = capsys.readouterr().out
    assert "SUMMARY: S203" in out
    assert "Trigger burst detected for S203" in out
    assert "Markers from .vmrk:" in out
    assert "Alignment: markers 4 -> 2" in out
    assert "Token map: {'token1': 'A', 'token2': 'B'}" in out


def test_summarize_one_file_warns_when_vmrk_and_subject_csv_are_missing(
    monkeypatch, tmp_path: Path, synthetic_raw, capsys
):
    subject_csv_dir = tmp_path / "subjects"
    subject_csv_dir.mkdir()

    raw_path = tmp_path / "S999.vhdr"
    raw_path.write_text("dummy", encoding="utf-8")

    args = _summary_args(subject_csv_dir)
    _patch_summary_dependencies(monkeypatch, synthetic_raw)

    cli.summarize_one_file(args, raw_path)

    out = capsys.readouterr().out
    assert ".vmrk file not found next to .vhdr" in out
    assert "Missing subject file for S999" in out
    assert "Exiting summary." in out


def test_summarize_one_file_reads_eeglab_files(monkeypatch, tmp_path: Path, synthetic_raw, capsys):
    subject_csv_dir = tmp_path / "subjects"
    subject_csv_dir.mkdir()

    raw_path = tmp_path / "S204.set"
    raw_path.write_text("dummy", encoding="utf-8")

    args = _summary_args(subject_csv_dir)
    args.behavioral_keep_codes = None
    _patch_summary_dependencies(monkeypatch, synthetic_raw)

    cli.summarize_one_file(args, raw_path)

    out = capsys.readouterr().out
    assert "VMRK file:" not in out
    assert "Missing subject file for S204" in out


def test_main_exits_early_for_one_file_summary(monkeypatch):
    called = {}

    monkeypatch.setattr(
        cli, "summarize_one_file", lambda args, path: called.setdefault("path", path)
    )

    cli.main(["--config", "config.yaml", "--summarize_one_file", "demo.vhdr"])

    assert called["path"] == Path("demo.vhdr")


def test_main_can_reprocess_before_plotting_when_metrics_are_missing(monkeypatch, tmp_path: Path, capsys):
    called = {}

    def fake_apply_preset(args, defaults):
        args._erp_core_preset_enabled = True
        args.reref = "tp9_tp10"
        args.l_freq = 0.1
        args.h_freq = 20.0
        args.volt_method = "simple"
        args.volt_auto_percentile = 97.5
        args.blink_auto_percentile = 99.0
        args.ica = "on"

    monkeypatch.setattr(cli, "apply_erp_core_preset", fake_apply_preset)
    monkeypatch.setattr(cli, "apply_config", lambda args, defaults: {"cfg": "ok"})
    monkeypatch.setattr(
        cli,
        "configure_gpu",
        lambda use_gpu, device=None: {
            "enabled": True,
            "backend": "cupy",
            "mne_cuda": "ok",
            "cupy": "ok",
        },
    )
    monkeypatch.setattr(cli, "capability_report", lambda: {"cuda": True})
    monkeypatch.setattr(cli, "format_capability_report", lambda rep: "GPU CAPABILITIES")
    monkeypatch.setattr(cli, "_prompt_yes_no", lambda msg: True)
    monkeypatch.setattr(
        cli,
        "run_full_pipeline",
        lambda args, defaults=None, cfg=None: called.setdefault(
            "run_full_pipeline",
            {
                "process_data": args.process_data,
                "get_metrics": args.get_metrics,
                "metrics": args.metrics,
                "cfg": cfg,
            },
        ),
    )
    monkeypatch.setattr(cli, "run_plot_figures", lambda args: called.setdefault("plot", args.plot_figures))

    cli.main(["--plot_figures", "--use_gpu", "--out_dir", str(tmp_path), "--config", "config.yaml"])

    out = capsys.readouterr().out
    assert "[ERP-CORE] preset enabled" in out
    assert "GPU CAPABILITIES" in out
    assert "[GPU] enabled" in out
    assert "Missing figure inputs:" in out
    assert called["run_full_pipeline"] == {
        "process_data": True,
        "get_metrics": True,
        "metrics": 1,
        "cfg": {"cfg": "ok"},
    }
    assert called["plot"] is True


def test_main_plot_figures_can_continue_without_reprocessing(monkeypatch, tmp_path: Path, capsys):
    called = {}

    monkeypatch.setattr(cli, "apply_erp_core_preset", lambda args, defaults: None)
    monkeypatch.setattr(cli, "apply_config", lambda args, defaults: {"cfg": "ok"})
    monkeypatch.setattr(
        cli,
        "configure_gpu",
        lambda use_gpu, device=None: {
            "enabled": False,
            "backend": "numpy",
            "mne_cuda": "disabled",
            "cupy": "disabled",
        },
    )
    monkeypatch.setattr(cli, "_prompt_yes_no", lambda msg: False)
    monkeypatch.setattr(
        cli,
        "run_full_pipeline",
        lambda *args, **kwargs: called.setdefault("run_full_pipeline", True),
    )
    monkeypatch.setattr(cli, "run_plot_figures", lambda args: called.setdefault("plot", args.plot_figures))

    cli.main(["--plot_figures", "--out_dir", str(tmp_path), "--config", "config.yaml"])

    out = capsys.readouterr().out
    assert "Proceeding with available metrics only." in out
    assert "run_full_pipeline" not in called
    assert called["plot"] is True
