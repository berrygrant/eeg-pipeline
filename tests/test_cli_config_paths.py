from pathlib import Path

import pytest

import eeg_pipeline.cli as cli


def _parser_and_args():
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--config", "config.yaml"])
    defaults = cli.build_defaults(parser)
    return parser, args, defaults


def _rich_config():
    return {
        "paths": {
            "raw_dir": Path("/tmp/raw"),
            "subject_csv_dir": Path("/tmp/subject_csv"),
            "out_dir": Path("/tmp/out"),
        },
        "channels": {
            "eog_chs": ["EOG"],
            "blink_proxy_chs": ["Fp1", "Fp2"],
            "drop_aux_chs": ["AUX1"],
        },
        "preprocess": {
            "montage": "standard_1020",
            "reref": "tp9_tp10",
            "l_freq": 0.5,
            "h_freq": 25.0,
            "notch_hz": [50.0, 60.0],
        },
        "events": {
            "standard_codes": [1],
            "deviant_codes": [2],
            "behavioral_keep_codes": [1, 2, 3],
            "drop_eeg_markers_by_gap_s": 1.5,
            "auto_drop_to_count": False,
            "condition_map": {"Oddball": [1], "Rare": [2]},
        },
        "epoching": {
            "tmin": -0.1,
            "tmax": 0.4,
            "baseline": [-0.1, 0.0],
        },
        "artifacts": {
            "test_window": [-0.05, 0.2],
            "max_reject_rate": 0.33,
            "blink": {
                "threshold_uv": 80.0,
                "win_ms": 150.0,
                "step_ms": 5.0,
                "auto_percentile": 97.0,
            },
            "voltage": {
                "pos_uv": 120.0,
                "neg_uv": -120.0,
                "method": "combined",
                "threshold_uv": 110.0,
                "win_ms": 120.0,
                "step_ms": 8.0,
                "step_uv_per_ms": 40.0,
                "auto_percentile": 96.0,
            },
        },
        "ica": {
            "mode": "auto",
            "auto_blink_rate_per_min": 12.0,
            "method": "infomax",
            "n_components": 15,
            "random_state": 11,
            "max_iter": 321,
            "fit_l_freq": 1.5,
            "fit_h_freq": 20.0,
            "decim": 2,
            "corr_thresh": 0.4,
            "max_exclude": 4,
            "save_ica": False,
        },
        "metrics": {
            "enabled": True,
            "channels": ["Fz", "Cz"],
            "conditions": ["Oddball", "Rare"],
            "difference_label": "DIFF",
            "compute_mmn": False,
            "compute_p300": True,
            "erp": {
                "enabled": True,
                "timeseries": True,
                "windows": [{"name": "W1", "tmin": 0.1, "tmax": 0.2}],
            },
            "tfr": {
                "enabled": True,
                "tmin": -0.05,
                "tmax": 0.3,
                "fmin": 2.0,
                "fmax": 8.0,
                "fstep": 2.0,
                "method": "morlet",
                "n_cycles_div": 5.0,
                "decim": 2,
                "time_decim": 3,
                "baseline": [-0.05, 0.0],
                "baseline_mode": "mean",
            },
        },
        "compute": {
            "use_gpu": True,
            "gpu_device": 2,
        },
        "labels": {
            "token_map": {"token1": "EH", "token2": "IH"},
        },
    }


def test_apply_config_maps_rich_config_onto_args(monkeypatch):
    _, args, defaults = _parser_and_args()
    monkeypatch.setattr(cli, "load_config", lambda path: _rich_config())

    cfg = cli.apply_config(args, defaults)

    assert cfg["paths"]["raw_dir"] == Path("/tmp/raw")
    assert args.raw_dir == Path("/tmp/raw")
    assert args.subject_csv_dir == Path("/tmp/subject_csv")
    assert args.out_dir == Path("/tmp/out")
    assert args.reref == "tp9_tp10"
    assert args.l_freq == 0.5
    assert args.h_freq == 25.0
    assert args.notch == [50.0, 60.0]
    assert args.eog_chs == ["EOG"]
    assert args.blink_proxy_chs == ["Fp1", "Fp2"]
    assert args.aux_chs == ["AUX1"]
    assert args.standard_codes == [1]
    assert args.deviant_codes == [2]
    assert args.behavioral_keep_codes == [1, 2, 3]
    assert args.drop_eeg_markers_by_gap_s == 1.5
    assert args.auto_drop_to_count == 0
    assert args.condition_map == {"Oddball": [1], "Rare": [2]}
    assert args.tmin == -0.1
    assert args.tmax == 0.4
    assert args.baseline == [-0.1, 0.0]
    assert args.art_test_tmin == -0.05
    assert args.art_test_tmax == 0.2
    assert args.blink_threshold_uv == 80.0
    assert args.volt_method == "combined"
    assert args.volt_threshold_uv == 110.0
    assert args.volt_step_uv_per_ms == 40.0
    assert args.max_reject_rate == 0.33
    assert args.ica == "auto"
    assert args.ica_method == "infomax"
    assert args.ica_n_components == "15"
    assert args.save_ica == 0
    assert args.metrics == 1
    assert args.metrics_erp_enabled is True
    assert args.metrics_tfr_enabled is True
    assert args.metrics_erp_timeseries is True
    assert args.metrics_channels == ["Fz", "Cz"]
    assert args.metrics_conditions == ["Oddball", "Rare"]
    assert args.erp_window == [["W1", 0.1, 0.2]]
    assert args.compute_mmn == 0
    assert args.difference_label == "DIFF"
    assert args.compute_p300 == 1
    assert args.tfr_tmin == -0.05
    assert args.tfr_tmax == 0.3
    assert args.tfr_fmin == 2.0
    assert args.tfr_fmax == 8.0
    assert args.tfr_fstep == 2.0
    assert args.tfr_method == "morlet"
    assert args.tfr_n_cycles_div == 5.0
    assert args.tfr_decim == 2
    assert args.tfr_time_decim == 3
    assert args.tfr_baseline == [-0.05, 0.0]
    assert args.tfr_baseline_mode == "mean"
    assert args.use_gpu is True
    assert args.gpu_device == 2
    assert args.token_map == ["token1=EH", "token2=IH"]


def test_apply_config_handles_missing_defaults_for_optional_voltage_fields(monkeypatch):
    _, args, defaults = _parser_and_args()
    monkeypatch.setattr(cli, "load_config", lambda path: _rich_config())

    for key in [
        "volt_method",
        "volt_threshold_uv",
        "volt_win_ms",
        "volt_step_ms",
        "volt_step_uv_per_ms",
        "volt_auto_percentile",
        "max_reject_rate",
    ]:
        defaults.pop(key, None)

    cli.apply_config(args, defaults)

    assert args.volt_method == "combined"
    assert args.volt_threshold_uv == 110.0
    assert args.volt_win_ms == 120.0
    assert args.volt_step_ms == 8.0
    assert args.volt_step_uv_per_ms == 40.0
    assert args.volt_auto_percentile == 96.0
    assert args.max_reject_rate == 0.33


def test_apply_erp_core_preset_sets_expected_defaults_only_when_enabled():
    _, args, defaults = _parser_and_args()
    args.erp_core = True

    cli.apply_erp_core_preset(args, defaults)

    assert args._erp_core_preset_enabled is True
    assert args.reref == "tp9_tp10"
    assert args.l_freq == 0.1
    assert args.h_freq == 20.0
    assert args.volt_method == "simple"
    assert args.volt_auto_percentile == 97.5
    assert args.blink_auto_percentile == 99.0
    assert args.ica == "on"


def test_run_full_pipeline_raises_when_no_raw_files(tmp_path: Path):
    _, args, defaults = _parser_and_args()
    args.raw_dir = tmp_path / "raw"
    args.subject_csv_dir = tmp_path / "subjects"
    args.out_dir = tmp_path / "out"
    args.raw_dir.mkdir()
    args.subject_csv_dir.mkdir()

    with pytest.raises(RuntimeError, match="No .vhdr or .set files found"):
        cli.run_full_pipeline(args, defaults=defaults, cfg={})


def test_run_full_pipeline_raises_when_subject_filter_matches_nothing(tmp_path: Path):
    _, args, defaults = _parser_and_args()
    args.raw_dir = tmp_path / "raw"
    args.subject_csv_dir = tmp_path / "subjects"
    args.out_dir = tmp_path / "out"
    args.raw_dir.mkdir()
    args.subject_csv_dir.mkdir()
    (args.raw_dir / "S001.vhdr").write_text("x", encoding="utf-8")
    args.subjects = ["S999"]

    with pytest.raises(RuntimeError, match="No matching raw files found"):
        cli.run_full_pipeline(args, defaults=defaults, cfg={})
