from argparse import Namespace
from pathlib import Path

import pytest

import eeg_pipeline.cli as cli


def test_set_if_default_only_updates_unmodified_arguments():
    args = Namespace(raw_dir="default", out_dir="custom")

    cli.set_if_default(args, {"raw_dir": "default", "out_dir": "default"}, "raw_dir", "/tmp/raw")
    cli.set_if_default(args, {"raw_dir": "default", "out_dir": "default"}, "out_dir", "/tmp/out")
    cli.set_if_default(args, {}, "missing", "ignored")

    assert args.raw_dir == "/tmp/raw"
    assert args.out_dir == "custom"


def test_bv_get_and_brainvision_links_ok_validate_referenced_files(tmp_path: Path):
    vhdr = tmp_path / "sample.vhdr"
    vmrk = tmp_path / "sample.vmrk"
    eeg = tmp_path / "sample.eeg"
    vmrk.write_text("marker", encoding="utf-8")
    eeg.write_text("data", encoding="utf-8")
    vhdr.write_text("MarkerFile=sample.vmrk\nDataFile=sample.eeg\n", encoding="utf-8")

    assert cli._bv_get(vhdr.read_text(encoding="utf-8"), "markerfile") == "sample.vmrk"
    assert cli.brainvision_links_ok(vhdr) == (True, "")

    vmrk.unlink()
    ok, reason = cli.brainvision_links_ok(vhdr)
    assert ok is False
    assert "MarkerFile=sample.vmrk" in reason


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (None, 0.99),
        ("20", 20),
        ("0.75", 0.75),
        (12, 12),
    ],
)
def test_parse_n_components_handles_common_input_types(raw_value, expected):
    assert cli._parse_n_components(raw_value) == expected


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("S203", "203"),
        ("203", "203"),
        ("subject-203", "203"),
    ],
)
def test_subject_number_from_stem_extracts_digits(stem, expected):
    assert cli.subject_number_from_stem(stem) == expected


def test_subject_number_from_stem_rejects_missing_digits():
    with pytest.raises(ValueError, match="Cannot parse subject number"):
        cli.subject_number_from_stem("subject")


def test_detect_trigger_bursts_reports_short_iti_and_dense_windows():
    diag = cli.detect_trigger_bursts(
        markers_pos=cli.np.array([0, 1, 2, 3, 100], dtype=int),
        sfreq=100.0,
        min_iti_s=0.02,
        burst_win_s=0.05,
        burst_count=4,
    )

    assert diag["burst_flag"] is True
    assert diag["n_short_iti"] == 3
    assert diag["burst_max_in_window"] == 4
    assert diag["burst_n_windows_ge_thresh"] >= 1


def test_detect_trigger_bursts_handles_short_inputs():
    diag = cli.detect_trigger_bursts(cli.np.array([10], dtype=int), sfreq=100.0)

    assert diag == {
        "burst_flag": False,
        "n_triggers": 1,
        "n_short_iti": 0,
        "min_iti_s": None,
        "burst_max_in_window": 1,
        "burst_n_windows_ge_thresh": 0,
    }


def test_figure_helpers_build_windows_and_choose_defaults():
    args = Namespace(
        figure_time_window=None,
        erp_window=[("Custom", "0.1", "0.2")],
        tmin=-0.2,
        tmax=0.6,
        figure_freq_band=None,
        tfr_fmin=2.0,
        tfr_fmax=8.0,
        compute_mmn=1,
        compute_p300=1,
    )

    assert cli._resolve_figure_time_window(args) == (0.1, 0.2)
    assert cli._resolve_figure_freq_band(args) == (2.0, 8.0)

    custom = cli._build_erp_windows(args)
    assert len(custom) == 1
    assert custom[0].name == "Custom"

    args.erp_window = None
    defaults = cli._build_erp_windows(args)
    assert [w.name for w in defaults] == ["MMN", "P300"]


def test_figure_helpers_cover_explicit_and_fallback_paths():
    args = Namespace(
        figure_time_window=("0.0", "0.3"),
        erp_window=None,
        tmin=-0.2,
        tmax=0.6,
        figure_freq_band=(4.0, 8.0),
        tfr_fmin=2.0,
        tfr_fmax=8.0,
        compute_mmn=0,
        compute_p300=0,
    )

    assert cli._resolve_figure_time_window(args) == (0.0, 0.3)
    assert cli._resolve_figure_freq_band(args) == (4.0, 8.0)
    assert cli._build_erp_windows(args) == []

    args.figure_time_window = None
    args.figure_freq_band = None
    assert cli._resolve_figure_time_window(args) == (-0.2, 0.6)
    assert cli._resolve_figure_freq_band(args) == (2.0, 8.0)


def test_prompt_yes_no_respects_tty_and_input(monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    assert cli._prompt_yes_no("ignored") is False

    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda msg: "YeS")
    assert cli._prompt_yes_no("ignored") is True


def test_prepare_output_dirs_and_parser_defaults(tmp_path: Path):
    out_dir = tmp_path / "derivatives"
    cli.prepare_output_dirs(out_dir)

    expected_dirs = {
        "01_clean_raw",
        "02_epochs",
        "03_evokeds",
        "04_grand_averages",
        "05_metrics",
        "00_ica",
    }
    assert {p.name for p in out_dir.iterdir()} == expected_dirs

    parser = cli.build_arg_parser()
    defaults = cli.build_defaults(parser)
    assert defaults["config"] is None
    assert defaults["process_data"] is False
    assert defaults["compute_mmn"] == 1
    assert defaults["tfr_method"] == "multitaper"


def test_main_defaults_to_processing_and_metrics(monkeypatch):
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
    monkeypatch.setattr(cli, "format_capability_report", lambda rep: "")
    monkeypatch.setattr(cli, "capability_report", lambda: {})
    monkeypatch.setattr(cli, "run_metrics_only", lambda args: called.setdefault("metrics_only", True))
    monkeypatch.setattr(cli, "run_plot_figures", lambda args: called.setdefault("plot", True))

    def fake_run_full_pipeline(args, defaults=None, cfg=None):
        called["process_data"] = args.process_data
        called["get_metrics"] = args.get_metrics
        called["metrics"] = args.metrics
        called["cfg"] = cfg

    monkeypatch.setattr(cli, "run_full_pipeline", fake_run_full_pipeline)

    cli.main(["--config", "config.yaml"])

    assert called == {
        "process_data": True,
        "get_metrics": True,
        "metrics": 1,
        "cfg": {"cfg": "ok"},
    }


def test_main_get_metrics_only_uses_metrics_runner(monkeypatch):
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
    monkeypatch.setattr(cli, "run_metrics_only", lambda args: called.setdefault("metrics_only", True))
    monkeypatch.setattr(cli, "run_full_pipeline", lambda *args, **kwargs: called.setdefault("process", True))
    monkeypatch.setattr(cli, "run_plot_figures", lambda args: called.setdefault("plot", True))

    cli.main(["--config", "config.yaml", "--get_metrics"])

    assert called == {"metrics_only": True}


def test_main_process_only_sets_metrics_zero_and_warns_when_gpu_falls_back(monkeypatch, capsys):
    called = {}

    monkeypatch.setattr(cli, "apply_erp_core_preset", lambda args, defaults: None)
    monkeypatch.setattr(cli, "apply_config", lambda args, defaults: {"cfg": "ok"})
    monkeypatch.setattr(
        cli,
        "configure_gpu",
        lambda use_gpu, device=None: {
            "enabled": False,
            "backend": "numpy",
            "mne_cuda": "missing",
            "cupy": "missing",
        },
    )
    monkeypatch.setattr(cli, "capability_report", lambda: {})
    monkeypatch.setattr(cli, "format_capability_report", lambda rep: "")

    def fake_run_full_pipeline(args, defaults=None, cfg=None):
        called["metrics"] = args.metrics
        called["cfg"] = cfg

    monkeypatch.setattr(cli, "run_full_pipeline", fake_run_full_pipeline)
    monkeypatch.setattr(cli, "run_metrics_only", lambda args: called.setdefault("metrics_only", True))

    cli.main(["--config", "config.yaml", "--process_data", "--use_gpu"])

    out = capsys.readouterr().out
    assert "[WARN] GPU requested but not available; falling back to CPU" in out
    assert called == {"metrics": 0, "cfg": {"cfg": "ok"}}
