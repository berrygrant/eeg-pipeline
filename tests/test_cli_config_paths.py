from pathlib import Path

import pytest

import eeg_pipeline.cli as cli
import eeg_pipeline.cli_config as cli_config
import eeg_pipeline.cli_pipeline as cli_pipeline


def _parser_and_args():
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--config", "config.yaml"])
    defaults = cli.build_defaults(parser)
    return parser, args, defaults


def _rich_config():
    return {
        "input": {
            "mode": "bids",
        },
        "task": "oddball",
        "paths": {
            "raw_dir": Path("/tmp/raw"),
            "subject_csv_dir": Path("/tmp/subjects"),
            "bids_root": Path("/tmp/bids"),
            "derivatives_root": Path("/tmp/derivatives"),
            "sourcedata_root": Path("/tmp/sourcedata"),
        },
        "bids": {
            "subjects": ["01"],
            "sessions": ["01"],
            "tasks": ["oddball"],
            "runs": ["01"],
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
            "csv_fallback_dir": Path("/tmp/fallback"),
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
        "conversion": {
            "enabled": False,
            "bids_output_root": Path("/tmp/converted_bids"),
            "overwrite": False,
        },
        "labels": {
            "token_map": {"token1": "EH", "token2": "IH"},
        },
    }


def test_apply_config_maps_rich_config_onto_args(monkeypatch):
    _, args, defaults = _parser_and_args()
    monkeypatch.setattr(cli_config, "load_config", lambda path, overrides=None: _rich_config())

    cfg = cli.apply_config(args, defaults)

    assert cfg["paths"]["bids_root"] == Path("/tmp/bids")
    assert args.input_mode == "bids"
    assert args.raw_dir == Path("/tmp/raw")
    assert args.subject_csv_dir == Path("/tmp/subjects")
    assert args.bids_root == Path("/tmp/bids")
    assert args.derivatives_root == Path("/tmp/derivatives")
    assert args.sourcedata_root == Path("/tmp/sourcedata")
    assert args.task_label == "oddball"
    assert args.subjects == ["01"]
    assert args.sessions == ["01"]
    assert args.tasks == ["oddball"]
    assert args.runs == ["01"]
    assert args.behavior_csv_fallback_dir == Path("/tmp/fallback")
    assert args.convert_to_bids is False
    assert args.conversion_bids_root == Path("/tmp/converted_bids")
    assert args.conversion_overwrite == 0
    assert args.reref == "tp9_tp10"
    assert args.standard_codes == [1]
    assert args.deviant_codes == [2]
    assert args.condition_map == {"Oddball": [1], "Rare": [2]}
    assert args.metrics == 1
    assert args.metrics_erp_enabled is True
    assert args.metrics_tfr_enabled is True
    assert args.metrics_erp_timeseries is True
    assert args.metrics_channels == ["Fz", "Cz"]
    assert args.metrics_conditions == ["Oddball", "Rare"]
    assert args.erp_window == [["W1", 0.1, 0.2]]
    assert args.token_map == ["token1=EH", "token2=IH"]


def test_apply_config_legacy_flag_overrides_config_mode(monkeypatch):
    _, args, defaults = _parser_and_args()
    args.legacy = True
    monkeypatch.setattr(cli_config, "load_config", lambda path, overrides=None: _rich_config())

    cli.apply_config(args, defaults)

    assert args.input_mode == "legacy"


def test_apply_config_legacy_flag_overrides_validation_mode(tmp_path: Path):
    cfg_path = tmp_path / "legacy.json"
    cfg_path.write_text(
        """
        {
          "paths": {},
          "events": {
            "standard_codes": [1],
            "deviant_codes": [2]
          }
        }
        """,
        encoding="utf-8",
    )

    parser = cli.build_arg_parser()
    args = parser.parse_args(["--config", str(cfg_path), "--legacy", "--raw_dir", "/tmp/legacy_raw"])
    defaults = cli.build_defaults(parser)

    cfg = cli.apply_config(args, defaults)

    assert cfg["input"]["mode"] == "legacy"
    assert cfg["paths"]["raw_dir"] == Path("/tmp/legacy_raw")
    assert args.input_mode == "legacy"
    assert args.raw_dir == Path("/tmp/legacy_raw")


def test_run_full_pipeline_raises_when_no_bids_recordings(tmp_path: Path):
    _, args, defaults = _parser_and_args()
    args.bids_root = tmp_path / "bids"
    args.derivatives_root = tmp_path / "derivatives"
    args.bids_root.mkdir()
    (args.bids_root / "dataset_description.json").write_text('{"Name":"Fixture","BIDSVersion":"1.11.1"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="No BIDS EEG recordings found"):
        cli.run_full_pipeline(args, defaults=defaults, cfg={})


def test_run_full_pipeline_respects_subject_filters(tmp_path: Path):
    _, args, defaults = _parser_and_args()
    args.bids_root = tmp_path / "bids"
    args.derivatives_root = tmp_path / "derivatives"
    eeg_dir = args.bids_root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (args.bids_root / "dataset_description.json").write_text('{"Name":"Fixture","BIDSVersion":"1.11.1"}', encoding="utf-8")
    (eeg_dir / "sub-01_task-oddball_run-01_eeg.vhdr").write_text("MarkerFile=sub.vmrk\nDataFile=sub.eeg\n", encoding="utf-8")
    (eeg_dir / "sub-01_task-oddball_run-01_events.tsv").write_text(
        "onset\tduration\tsample\ttrial_type\tvalue\n0.0\t0.1\t100\tStandard\t1\n",
        encoding="utf-8",
    )
    args.subjects = ["sub-99"]

    with pytest.raises(RuntimeError, match="No BIDS EEG recordings found"):
        cli.run_full_pipeline(args, defaults=defaults, cfg={})


def _bids_fixture_with_one_recording(tmp_path: Path):
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--config", "config.yaml"])
    defaults = cli.build_defaults(parser)
    args.bids_root = tmp_path / "bids"
    args.derivatives_root = tmp_path / "derivatives"
    eeg_dir = args.bids_root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (args.bids_root / "dataset_description.json").write_text(
        '{"Name":"Fixture","BIDSVersion":"1.11.1"}', encoding="utf-8"
    )
    (eeg_dir / "sub-01_task-oddball_run-01_eeg.vhdr").write_text(
        "MarkerFile=sub.vmrk\nDataFile=sub.eeg\n", encoding="utf-8"
    )
    (eeg_dir / "sub-01_task-oddball_run-01_events.tsv").write_text(
        "onset\tduration\tsample\ttrial_type\tvalue\n0.0\t0.1\t100\tStandard\t1\n",
        encoding="utf-8",
    )
    return args, defaults


def test_run_full_pipeline_aggregates_by_default(monkeypatch, tmp_path: Path):
    args, defaults = _bids_fixture_with_one_recording(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(cli_pipeline, "_process_recording", lambda rec, **kw: None)
    monkeypatch.setattr(cli_pipeline, "run_aggregation", lambda root, a: calls.append("ran"))

    cli.run_full_pipeline(args, defaults=defaults, cfg={})

    assert calls == ["ran"]


def test_run_full_pipeline_skips_aggregation_with_skip_aggregate(monkeypatch, tmp_path: Path, capsys):
    """--skip_aggregate must suppress the dataset-level rebuild.

    Per-subject outputs go to distinct paths, but the dataset-level tables and
    grand averages are shared. If every concurrent task aggregated at the end,
    N array tasks would race on those shared paths and read files other tasks
    were still writing -- the final state would be whichever finished last.
    """
    args, defaults = _bids_fixture_with_one_recording(tmp_path)
    args.skip_aggregate = True
    calls: list[str] = []
    monkeypatch.setattr(cli_pipeline, "_process_recording", lambda rec, **kw: None)
    monkeypatch.setattr(cli_pipeline, "run_aggregation", lambda root, a: calls.append("ran"))

    cli.run_full_pipeline(args, defaults=defaults, cfg={})

    assert calls == []
    assert "--aggregate_only" in capsys.readouterr().out
