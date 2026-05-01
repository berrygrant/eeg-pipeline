from pathlib import Path

import pytest

from eeg_pipeline.oneclick import backend


def test_stage_args_defaults_to_process_and_metrics():
    assert backend._stage_args(None) == ["--process_data", "--get_metrics"]


def test_stage_args_uses_selected_stages():
    assert backend._stage_args({"processData": False, "getMetrics": True, "plotFigures": True}) == [
        "--get_metrics",
        "--plot_figures",
    ]


def test_stage_args_rejects_empty_selection():
    with pytest.raises(ValueError, match="At least one stage"):
        backend._stage_args({"processData": False, "getMetrics": False, "plotFigures": False})


def test_validate_config_reports_summary(tmp_path: Path):
    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
input:
  mode: bids
paths:
  bids_root: {bids_root}
events:
  behavioral_keep_codes: [1, 2]
  standard_codes: [1]
  deviant_codes: [2]
""",
        encoding="utf-8",
    )

    result = backend.validate_config(cfg_path)

    assert result["ok"] is True
    assert result["summary"]["inputMode"] == "bids"
    assert result["summary"]["bidsRoot"] == bids_root

