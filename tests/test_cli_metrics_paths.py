from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import eeg_pipeline.cli as cli
import eeg_pipeline.cli_metrics as cli_metrics


def _base_metrics_args(bids_root: Path, derivatives_root: Path) -> Namespace:
    return Namespace(
        bids_root=str(bids_root),
        derivatives_root=str(derivatives_root),
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
        montage="standard_1020",
        reref="average",
        l_freq=0.1,
        h_freq=20.0,
        notch=[60.0],
        art_test_tmin=-0.1,
        art_test_tmax=0.2,
        blink_threshold_uv=75.0,
        blink_win_ms=200.0,
        blink_step_ms=10.0,
        blink_auto_percentile=None,
        volt_pos_uv=150.0,
        volt_neg_uv=-150.0,
        volt_threshold_uv=150.0,
        volt_win_ms=200.0,
        volt_step_ms=10.0,
        volt_step_uv_per_ms=None,
        volt_auto_percentile=None,
        volt_method="simple",
        max_reject_rate=None,
        ica="off",
        ica_method="fastica",
        ica_n_components="0.99",
        ica_random_state=97,
        ica_max_iter=512,
        ica_fit_l_freq=1.0,
        ica_fit_h_freq=None,
        ica_decim=3,
        ica_corr_thresh=0.3,
        ica_max_exclude=3,
    )


def test_subject_from_epochs_path_uses_bids_entities():
    assert cli._subject_from_epochs_path(Path("sub-001_task-oddball_epo.fif")) == "sub-001_task-oddball_eeg"


def test_run_metrics_only_raises_when_no_epoch_files(tmp_path: Path):
    bids_root = tmp_path / "bids"
    derivatives_root = tmp_path / "derivatives"
    bids_root.mkdir()
    derivatives_root.mkdir()
    args = _base_metrics_args(bids_root, derivatives_root)

    with pytest.raises(RuntimeError, match="No epochs found"):
        cli.run_metrics_only(args)


def test_run_metrics_only_writes_combined_outputs(monkeypatch, tmp_path: Path, synthetic_epochs):
    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    (bids_root / "dataset_description.json").write_text('{"Name":"Fixture","BIDSVersion":"1.11.1"}', encoding="utf-8")
    derivatives_root = tmp_path / "derivatives"
    dataset_root = cli_metrics._prepare_derivatives_root(
        Namespace(bids_root=str(bids_root), derivatives_root=str(derivatives_root))
    )
    epoch_path = dataset_root / "sub-001" / "eeg" / "sub-001_task-oddball_epo.fif"
    epoch_path.parent.mkdir(parents=True, exist_ok=True)
    epoch_path.write_text("epo", encoding="utf-8")

    args = _base_metrics_args(bids_root, derivatives_root)
    args.metrics_erp_timeseries = True
    args.metrics_channels = None
    args.condition_map = {"Oddball": 1}

    monkeypatch.setattr(cli_metrics, "_build_erp_windows", lambda args_obj: [cli_metrics.ERP_WINDOWS["MMN"]])
    monkeypatch.setattr(cli_metrics, "load_epochs", lambda path: SimpleNamespace(epochs=synthetic_epochs))
    monkeypatch.setattr(cli_metrics.mne, "pick_types", lambda info, eeg=True, eog=False: [0, 1])
    monkeypatch.setattr(
        cli_metrics,
        "compute_erp_metrics",
        lambda *args_in, **kwargs: pd.DataFrame([{"subject": kwargs["subject"], "status": "OK"}]),
    )
    monkeypatch.setattr(
        cli_metrics,
        "compute_erp_timeseries",
        lambda *args_in, **kwargs: pd.DataFrame([{"subject": kwargs["subject"], "status": "OK"}]),
    )
    monkeypatch.setattr(
        cli_metrics,
        "compute_tfr_metrics",
        lambda *args_in, **kwargs: pd.DataFrame([{"subject": kwargs["subject"], "status": "OK"}]),
    )

    cli.run_metrics_only(args)

    metrics_root = dataset_root / "eeg"
    subject_root = dataset_root / "sub-001" / "eeg"
    assert (subject_root / "sub-001_task-oddball_desc-erp_metrics.tsv").exists()
    assert (subject_root / "sub-001_task-oddball_desc-tfr_metrics.tsv").exists()
    assert (subject_root / "sub-001_task-oddball_desc-erp_timeseries.parquet").exists()
    assert (metrics_root / "desc-erp_metrics.tsv").exists()
    assert (metrics_root / "desc-tfr_metrics.tsv").exists()
    assert (metrics_root / "desc-erp_timeseries.parquet").exists()
