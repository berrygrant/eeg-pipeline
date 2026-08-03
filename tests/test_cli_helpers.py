from argparse import Namespace
from pathlib import Path

import pytest

import eeg_pipeline.cli as cli
import eeg_pipeline.cli_common as cli_common
import eeg_pipeline.cli_pipeline as cli_pipeline


def test_run_full_pipeline_does_not_alias_conversion_into_recursion(monkeypatch):
    """Regression: --legacy --convert_to_bids --process_data must not recurse.

    cli.run_full_pipeline previously repointed cli_pipeline's own
    run_legacy_to_bids_conversion at the cli wrapper, whose body then called
    that same attribute -> unbounded recursion. Guard that the name resolved
    inside run_full_pipeline stays the real in-module implementation.
    """
    original = cli_pipeline.run_legacy_to_bids_conversion
    captured = {}

    def fake_run_full_pipeline(args, defaults=None, cfg=None):
        captured["conversion"] = cli_pipeline.run_legacy_to_bids_conversion
        return []

    monkeypatch.setattr(cli_pipeline, "run_full_pipeline", fake_run_full_pipeline)

    cli.run_full_pipeline(Namespace(), defaults={}, cfg={})

    assert captured["conversion"] is original
    assert cli_pipeline.run_legacy_to_bids_conversion is original


def test_set_if_default_only_updates_unmodified_arguments():
    args = Namespace(bids_root="default", derivatives_root="custom")

    cli_common.set_if_default(args, {"bids_root": "default", "derivatives_root": "default"}, "bids_root", "/tmp/bids")
    cli_common.set_if_default(args, {"bids_root": "default", "derivatives_root": "default"}, "derivatives_root", "/tmp/derivatives")
    cli_common.set_if_default(args, {}, "missing", "ignored")

    assert args.bids_root == "/tmp/bids"
    assert args.derivatives_root == "custom"


def test_bv_get_and_brainvision_links_ok_validate_referenced_files(tmp_path: Path):
    vhdr = tmp_path / "sample.vhdr"
    vmrk = tmp_path / "sample.vmrk"
    eeg = tmp_path / "sample.eeg"
    vmrk.write_text("marker", encoding="utf-8")
    eeg.write_text("data", encoding="utf-8")
    vhdr.write_text("MarkerFile=sample.vmrk\nDataFile=sample.eeg\n", encoding="utf-8")

    assert cli_common._bv_get(vhdr.read_text(encoding="utf-8"), "markerfile") == "sample.vmrk"
    assert cli_common.brainvision_links_ok(vhdr) == (True, "")

    vmrk.unlink()
    ok, reason = cli_common.brainvision_links_ok(vhdr)
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
    assert cli_common._parse_n_components(raw_value) == expected


def test_detect_trigger_bursts_reports_short_iti_and_dense_windows():
    diag = cli_common.detect_trigger_bursts(
        markers_pos=cli_common.np.array([0, 1, 2, 3, 100], dtype=int),
        sfreq=100.0,
        min_iti_s=0.02,
        burst_win_s=0.05,
        burst_count=4,
    )

    assert diag["burst_flag"] is True
    assert diag["n_short_iti"] == 3
    assert diag["burst_max_in_window"] == 4
    assert diag["burst_n_windows_ge_thresh"] >= 1


def test_prepare_output_dirs_creates_derivative_dataset(tmp_path: Path):
    derivatives_root = tmp_path / "derivatives"
    cli.prepare_output_dirs(derivatives_root)

    dataset_description = derivatives_root / "eeg-pipeline" / "dataset_description.json"
    assert dataset_description.exists()


def test_subject_from_epochs_path_uses_source_entities():
    path = Path("sub-001_task-oddball_run-01_epo.fif")
    assert cli._subject_from_epochs_path(path) == "sub-001_task-oddball_run-01_eeg"


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


def test_main_convert_only_runs_conversion_stage(monkeypatch):
    called = {}

    monkeypatch.setattr(cli, "apply_erp_core_preset", lambda args, defaults: None)

    def fake_apply_config(args, defaults):
        args.input_mode = "legacy"
        return {"input": {"mode": "legacy"}}

    monkeypatch.setattr(cli, "apply_config", fake_apply_config)
    monkeypatch.setattr(cli, "_finalize_runtime_paths", lambda args, cfg=None: None)
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
    monkeypatch.setattr(
        cli,
        "run_legacy_to_bids_conversion",
        lambda args, defaults=None, cfg=None: called.setdefault("converted", True),
    )
    monkeypatch.setattr(cli, "run_full_pipeline", lambda *args, **kwargs: called.setdefault("processed", True))

    cli.main(["--config", "config.yaml", "--legacy", "--convert_to_bids"])

    assert called == {"converted": True}


def test_stage_timings_accumulate_and_render_qc_columns():
    timings = cli_common.StageTimings()

    with timings.stage("preprocess"):
        pass
    # Re-entering a stage must sum into one figure, since ICA is timed at three
    # separate call sites (diagnostics, fit, apply) but reports as one stage.
    with timings.stage("ica"):
        pass
    with timings.stage("ica"):
        pass

    columns = timings.as_qc_columns()
    assert set(columns) == {"t_preprocess_s", "t_ica_s"}
    assert all(isinstance(v, float) and v >= 0.0 for v in columns.values())


def test_stage_timings_record_elapsed_even_when_stage_raises():
    timings = cli_common.StageTimings()

    # Timing must never swallow or mask a failure, and a run that dies partway
    # is exactly the one worth profiling -- so the elapsed time is still kept.
    with pytest.raises(ValueError, match="boom"):
        with timings.stage("metrics"):
            raise ValueError("boom")

    assert "t_metrics_s" in timings.as_qc_columns()


def test_process_recording_stamps_stage_timings_onto_every_qc_row(monkeypatch):
    """_process_recording merges timings into each row its stage work appended."""

    def fake_stages(recording, *, timings, rows, **kwargs):
        with timings.stage("preprocess"):
            pass
        rows.append({"subject": "sub-01", "status": "OK"})
        rows.append({"subject": "sub-01", "status": "SKIP_SOMETHING"})

    monkeypatch.setattr(cli_pipeline, "_process_recording_stages", fake_stages)

    rows = [{"subject": "sub-00", "status": "PRE_EXISTING"}]
    cli_pipeline._process_recording(object(), rows=rows)

    # Rows appended by this call get timings; rows from earlier calls are untouched.
    assert "t_preprocess_s" not in rows[0]
    assert "t_preprocess_s" in rows[1]
    assert "t_preprocess_s" in rows[2]


def test_process_recording_stamps_timings_even_when_stages_raise(monkeypatch):
    def failing_stages(recording, *, timings, rows, **kwargs):
        with timings.stage("preprocess"):
            pass
        rows.append({"subject": "sub-01", "status": "PARTIAL"})
        raise RuntimeError("stage exploded")

    monkeypatch.setattr(cli_pipeline, "_process_recording_stages", failing_stages)

    rows = []
    with pytest.raises(RuntimeError, match="stage exploded"):
        cli_pipeline._process_recording(object(), rows=rows)

    assert rows and "t_preprocess_s" in rows[0]
