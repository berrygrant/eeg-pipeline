from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import eeg_pipeline.cli as cli


def _base_metrics_args(out_dir: Path) -> Namespace:
    return Namespace(
        out_dir=str(out_dir),
        metrics_erp_enabled=True,
        metrics_tfr_enabled=True,
        metrics_channels=["Fz"],
        metrics_conditions=None,
        condition_map=None,
        compute_mmn=1,
        compute_p300=0,
        difference_label=None,
        metrics_erp_timeseries=False,
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


def test_subject_from_epochs_path_trims_epo_suffix():
    assert cli._subject_from_epochs_path(Path("sub-001-epo.fif")) == "sub-001"
    assert cli._subject_from_epochs_path(Path("sub-001.fif")) == "sub-001"


def test_run_metrics_only_raises_when_no_epoch_files(tmp_path: Path):
    args = _base_metrics_args(tmp_path)

    with pytest.raises(RuntimeError, match="No epochs found"):
        cli.run_metrics_only(args)


def test_run_metrics_only_warns_when_both_metrics_disabled(tmp_path: Path, capsys):
    epochs_dir = tmp_path / "02_epochs"
    epochs_dir.mkdir(parents=True)
    (epochs_dir / "sub-001-epo.fif").touch()
    args = _base_metrics_args(tmp_path)
    args.metrics_erp_enabled = False
    args.metrics_tfr_enabled = False

    cli.run_metrics_only(args)

    out = capsys.readouterr().out
    assert "both ERP and TFR are disabled" in out


def test_run_metrics_only_writes_combined_outputs(monkeypatch, tmp_path: Path, synthetic_epochs):
    epochs_dir = tmp_path / "02_epochs"
    epochs_dir.mkdir(parents=True)
    (epochs_dir / "sub-001-epo.fif").touch()

    args = _base_metrics_args(tmp_path)
    args.metrics_erp_timeseries = True
    args.metrics_channels = None
    args.condition_map = {"Oddball": 1}

    monkeypatch.setattr(cli, "_build_erp_windows", lambda args_obj: [cli.ERP_WINDOWS["MMN"]])
    monkeypatch.setattr(cli, "load_epochs", lambda path: SimpleNamespace(epochs=synthetic_epochs))
    monkeypatch.setattr(cli.mne, "pick_types", lambda info, eeg=True, eog=False: [0, 1])
    monkeypatch.setattr(
        cli,
        "compute_erp_metrics",
        lambda *args_in, **kwargs: pd.DataFrame([{"subject": kwargs["subject"], "status": "OK"}]),
    )
    monkeypatch.setattr(
        cli,
        "compute_erp_timeseries",
        lambda *args_in, **kwargs: pd.DataFrame([{"subject": kwargs["subject"], "status": "OK"}]),
    )
    monkeypatch.setattr(
        cli,
        "compute_tfr_metrics",
        lambda *args_in, **kwargs: pd.DataFrame([{"subject": kwargs["subject"], "status": "OK"}]),
    )

    cli.run_metrics_only(args)

    metrics_dir = tmp_path / "05_metrics"
    assert (metrics_dir / "sub-001_erp_metrics.csv").exists()
    assert (metrics_dir / "sub-001_tfr_metrics.csv").exists()
    assert (metrics_dir / "erp_metrics_all.csv").exists()
    assert (metrics_dir / "tfr_metrics_all.csv").exists()
    assert (metrics_dir / "erp_timeseries_all.parquet").exists()
    assert (metrics_dir / "erp_timeseries" / "sub-001_erp_timeseries.parquet").exists()
