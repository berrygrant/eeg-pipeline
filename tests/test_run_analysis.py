from argparse import Namespace

import matplotlib
import mne
import numpy as np

import run_analysis
from eeg_pipeline.metrics.erp_timeseries import ERP_TIMESERIES_COLUMNS, compute_erp_timeseries

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _make_epochs(scale: float) -> mne.Epochs:
    sfreq = 100.0
    ch_names = ["Fz", "Cz"]
    info = mne.create_info(ch_names, sfreq, ch_types=["eeg", "eeg"])
    info.set_montage("standard_1020")

    tmin = -0.2
    n_times = 200
    times = np.arange(n_times) / sfreq + tmin
    event_codes = np.array([1, 2, 1, 2], dtype=int)
    data = np.zeros((len(event_codes), len(ch_names), n_times), dtype=float)

    for idx, code in enumerate(event_codes):
        base = np.sin(2 * np.pi * 10 * times)
        effect = scale if code == 1 else 2 * scale
        data[idx, 0, :] = effect * base
        data[idx, 1, :] = effect * np.cos(2 * np.pi * 6 * times)

    events = np.column_stack(
        [
            np.arange(0, len(event_codes) * 250, 250, dtype=int),
            np.zeros(len(event_codes), dtype=int),
            event_codes,
        ]
    )
    return mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"Standard": 1, "Deviant": 2},
        tmin=tmin,
        baseline=None,
        verbose="error",
    )


def test_maybe_make_figures_saves_expected_outputs_and_closes_figures(tmp_path):
    epochs_dir = tmp_path / "epochs"
    out_dir = tmp_path / "out"
    epochs_dir.mkdir()
    out_dir.mkdir()

    for idx, subject in enumerate(("s001", "s002"), start=1):
        _make_epochs(idx * 1e-6).save(
            epochs_dir / f"{subject}-epo.fif",
            overwrite=True,
            verbose="error",
        )

    args = Namespace(
        make_figures=True,
        do_erp=True,
        do_tfr=True,
        epochs_dir=str(epochs_dir),
        pattern="*-epo.fif",
        conditions=["Standard", "Deviant"],
        channels=["Fz", "Cz"],
        fig_format="png",
        dpi=72,
        tfr_fmin=8.0,
        tfr_fmax=12.0,
        tfr_fstep=2.0,
        tfr_method="morlet",
        tfr_n_cycles_div=10.0,
        tfr_decim=1,
        tfr_baseline=[-0.2, 0.0],
        tfr_baseline_mode="logratio",
    )

    plt.close("all")
    run_analysis._maybe_make_figures(out_dir=out_dir, erp_rows=[], tfr_rows=[], args=args)

    figure_names = {path.name for path in (out_dir / "figures").glob("*.png")}
    assert {
        "erp_Standard.png",
        "erp_joint_Standard.png",
        "erp_Deviant.png",
        "erp_joint_Deviant.png",
        "tfr_Standard.png",
        "tfr_Deviant.png",
    } <= figure_names
    assert plt.get_fignums() == []


def test_erp_timeseries_empty_epochs_uses_stable_schema():
    df = compute_erp_timeseries(None, subject="s001", channels=["Fz"])

    assert list(df.columns) == ERP_TIMESERIES_COLUMNS
    assert df.loc[0, "status"] == "EMPTY_EPOCHS"
