import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from eeg_pipeline.config import (
    _apply_defaults,
    _as_float_list,
    _as_int_list,
    _normalize_condition_map,
    _normalize_config,
    _normalize_token_map,
    _parse_n_components,
    _read_config_file,
    _validate_config,
    config_get,
    load_config,
)


def _valid_min_cfg():
    return {
        "paths": {
            "bids_root": "/tmp/bids",
            "derivatives_root": "/tmp/derivatives",
        },
        "events": {
            "standard_codes": [1],
            "deviant_codes": [2],
            "behavioral_keep_codes": [1, 2],
        },
    }


def test_load_config_json_applies_defaults_and_normalizes_types(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "paths": {
                    "bids_root": "/tmp/bids",
                    "derivatives_root": "/tmp/derivatives",
                },
                "bids": {
                    "subjects": ["01"],
                    "tasks": "oddball",
                },
                "events": {
                    "standard_codes": ["110"],
                    "deviant_codes": ["111"],
                    "behavioral_keep_codes": ["110", "111"],
                    "csv_fallback_dir": "/tmp/fallback",
                },
                "ica": {"n_components": "20"},
                "labels": {"token_map": ["Token1=EH", "Token2=IH"]},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg["paths"]["bids_root"] == Path("/tmp/bids")
    assert cfg["paths"]["derivatives_root"] == Path("/tmp/derivatives")
    assert cfg["events"]["csv_fallback_dir"] == Path("/tmp/fallback")
    assert cfg["bids"]["subjects"] == ["01"]
    assert cfg["bids"]["tasks"] == ["oddball"]
    assert cfg["events"]["standard_codes"] == [110]
    assert cfg["events"]["deviant_codes"] == [111]
    assert cfg["ica"]["n_components"] == 20
    assert cfg["labels"]["token_map"] == {"token1": "EH", "token2": "IH"}
    assert cfg["metrics"]["erp"]["enabled"] is True


def test_load_config_rejects_missing_non_mapping_and_empty_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(tmp_path / "missing.json")

    list_cfg = tmp_path / "list.json"
    list_cfg.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Top-level config must be"):
        load_config(list_cfg)

    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_config(empty_yaml)
    assert "Provide at least one input path" in str(exc_info.value)


def test_load_config_accepts_condition_map_without_standard_deviant_codes(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "paths": {
                    "bids_root": "/tmp/bids",
                    "derivatives_root": "/tmp/derivatives",
                },
                "events": {
                    "condition_map": {"Standard": [1], "Deviant": [2]},
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg["events"]["condition_map"] == {"Standard": [1], "Deviant": [2]}


def test_load_config_applies_overrides_before_validation(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "paths": {},
                "events": {
                    "standard_codes": [1],
                    "deviant_codes": [2],
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(
        cfg_path,
        overrides={
            "input": {"mode": "legacy"},
            "paths": {"raw_dir": "/tmp/raw"},
        },
    )

    assert cfg["input"]["mode"] == "legacy"
    assert cfg["paths"]["raw_dir"] == Path("/tmp/raw")


def test_config_get_returns_default_for_missing_paths():
    cfg = {"events": {"standard_codes": [110]}}

    assert config_get(cfg, "events.standard_codes") == [110]
    assert config_get(cfg, "events.deviant_codes", default=[]) == []


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (None, 0.99),
        ("", 0.99),
        ("20", 20),
        ("0.75", 0.75),
    ],
)
def test_parse_n_components_handles_default_int_and_float_values(raw_value, expected):
    assert _parse_n_components(raw_value) == expected


def test_normalize_token_map_supports_dict_and_shorthand_list_inputs():
    assert _normalize_token_map({"Token1": "EH", "token2": "IH"}) == {
        "token1": "EH",
        "token2": "IH",
    }
    assert _normalize_token_map(["EH", "IH"]) == {"token1": "EH", "token2": "IH"}
    assert _normalize_token_map(["ignored", "Token1=EH", "Token2=IH"]) == {
        "token1": "EH",
        "token2": "IH",
    }
    assert _normalize_token_map("") is None
    assert _normalize_token_map({"other": "ignored"}) is None
    assert _normalize_token_map("Token1=EH") is None


def test_parse_n_components_surfaces_invalid_values():
    with pytest.raises(ValueError):
        _parse_n_components("not-a-number")


def test_read_config_file_supports_yaml_and_rejects_unknown_extensions(monkeypatch, tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("ignored", encoding="utf-8")

    fake_yaml = ModuleType("yaml")
    fake_yaml.safe_load = lambda fh: {"task": "yaml"}
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)

    assert _read_config_file(yaml_path) == {"task": "yaml"}

    bad_path = tmp_path / "config.txt"
    bad_path.write_text("ignored", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported config extension"):
        _read_config_file(bad_path)


def test_apply_defaults_and_normalize_config_convert_optional_values():
    cfg = _apply_defaults(_valid_min_cfg())
    cfg["input"]["mode"] = "LEGACY"
    cfg["paths"]["raw_dir"] = "/tmp/raw"
    cfg["paths"]["subject_csv_dir"] = "/tmp/subjects"
    cfg["paths"]["sourcedata_root"] = "/tmp/sourcedata"
    cfg["bids"]["subjects"] = "01"
    cfg["bids"]["sessions"] = ["01", "02"]
    cfg["events"]["csv_fallback_dir"] = "/tmp/fallback"
    cfg["events"]["condition_map"] = {"Oddball": "7", "Rare": [8]}
    cfg["artifacts"]["max_reject_rate"] = "null"
    cfg["artifacts"]["blink"]["auto_percentile"] = "None"
    cfg["artifacts"]["voltage"]["method"] = "Combined"
    cfg["artifacts"]["voltage"]["threshold_uv"] = "125.0"
    cfg["artifacts"]["voltage"]["win_ms"] = "150.0"
    cfg["artifacts"]["voltage"]["step_ms"] = "8.0"
    cfg["artifacts"]["voltage"]["step_uv_per_ms"] = "None"
    cfg["artifacts"]["voltage"]["auto_percentile"] = "95.5"
    cfg["ica"]["mode"] = "AUTO"
    cfg["ica"]["auto_blink_rate_per_min"] = "12.0"
    cfg["ica"]["n_components"] = "1e-1"
    cfg["ica"]["random_state"] = "7"
    cfg["ica"]["max_iter"] = "256"
    cfg["ica"]["fit_l_freq"] = "1.5"
    cfg["ica"]["fit_h_freq"] = "None"
    cfg["ica"]["decim"] = "4"
    cfg["ica"]["corr_thresh"] = "0.4"
    cfg["ica"]["max_exclude"] = "5"
    cfg["ica"]["save_ica"] = 0
    cfg["labels"]["token_map"] = "EH IH"
    cfg["metrics"]["erp"]["conditions"] = "Oddball"
    cfg["metrics"]["erp"]["windows"] = [{"name": "N2", "tmin": "-0.2", "tmax": "0.3"}]
    cfg["compute"]["use_gpu"] = 1
    cfg["compute"]["gpu_device"] = ""
    cfg["conversion"]["enabled"] = 1
    cfg["conversion"]["bids_output_root"] = "/tmp/converted_bids"
    cfg["conversion"]["overwrite"] = 0

    normalized = _normalize_config(cfg)

    assert normalized["input"]["mode"] == "legacy"
    assert normalized["paths"]["bids_root"] == Path("/tmp/bids")
    assert normalized["paths"]["raw_dir"] == Path("/tmp/raw")
    assert normalized["paths"]["subject_csv_dir"] == Path("/tmp/subjects")
    assert normalized["paths"]["sourcedata_root"] == Path("/tmp/sourcedata")
    assert normalized["events"]["csv_fallback_dir"] == Path("/tmp/fallback")
    assert normalized["bids"]["subjects"] == ["01"]
    assert normalized["bids"]["sessions"] == ["01", "02"]
    assert normalized["events"]["condition_map"] == {"Oddball": [7], "Rare": [8]}
    assert normalized["artifacts"]["max_reject_rate"] is None
    assert normalized["artifacts"]["blink"]["auto_percentile"] is None
    assert normalized["artifacts"]["voltage"]["method"] == "combined"
    assert normalized["artifacts"]["voltage"]["threshold_uv"] == 125.0
    assert normalized["artifacts"]["voltage"]["step_uv_per_ms"] is None
    assert normalized["artifacts"]["voltage"]["auto_percentile"] == 95.5
    assert normalized["ica"]["mode"] == "auto"
    assert normalized["ica"]["n_components"] == 0.1
    assert normalized["ica"]["fit_h_freq"] is None
    assert normalized["ica"]["save_ica"] is False
    assert normalized["labels"]["token_map"] == {"token1": "EH", "token2": "IH"}
    assert normalized["metrics"]["erp"]["conditions"] == ["Oddball"]
    assert normalized["metrics"]["erp"]["windows"] == [{"name": "N2", "tmin": "-0.2", "tmax": "0.3"}]
    assert normalized["compute"]["use_gpu"] is True
    assert normalized["compute"]["gpu_device"] is None
    assert normalized["conversion"]["enabled"] is True
    assert normalized["conversion"]["bids_output_root"] == Path("/tmp/converted_bids")
    assert normalized["conversion"]["overwrite"] is False


def test_scalar_list_helpers_and_condition_map_normalization_cover_edge_cases():
    assert _as_int_list(None) == []
    assert _as_int_list("5") == [5]
    assert _as_float_list(None) == []
    assert _as_float_list("1.25") == [1.25]
    assert _normalize_condition_map(None) is None
    assert _as_int_list((1, "2")) == [1, 2]
    assert _as_float_list((1, "2.5")) == [1.0, 2.5]
    assert _normalize_condition_map({"A": "1", "B": [2, 3]}) == {"A": [1], "B": [2, 3]}

    with pytest.raises(ValueError, match="mapping of name"):
        _normalize_condition_map(["bad"])


def test_validate_config_collects_multiple_readable_errors():
    cfg = _apply_defaults({"paths": {}, "events": {}})
    cfg["events"]["standard_codes"] = ["x"]
    cfg["events"]["deviant_codes"] = [1]
    cfg["events"]["behavioral_keep_codes"] = ["bad"]
    cfg["events"]["condition_map"] = {"A": [1, 2]}
    cfg["epoching"]["baseline"] = [0.0]
    cfg["ica"]["mode"] = "bad"
    cfg["preprocess"]["reref"] = "bad"
    cfg["artifacts"]["voltage"]["method"] = "bad"
    cfg["artifacts"]["blink"]["auto_percentile"] = "101"
    cfg["artifacts"]["voltage"]["auto_percentile"] = "oops"
    cfg["metrics"]["erp"]["windows"] = [{"name": "W"}]

    with pytest.raises(ValueError) as exc_info:
        _validate_config(cfg)

    msg = str(exc_info.value)
    assert "Provide at least one input path" in msg
    assert "input.mode='bids' requires paths.bids_root" in msg
    assert "epoching.baseline must be a 2-item list" in msg
    assert "ica.mode must be one of" in msg
    assert "preprocess.reref must be one of" in msg
    assert "artifacts.voltage.method must be one of" in msg
    assert "events.condition_map['A'] must map to a single code" in msg


def test_validate_config_catches_condition_map_and_keep_code_conflicts():
    cfg = _apply_defaults(
        {
            "input": {"mode": "legacy"},
            "paths": {"raw_dir": "/tmp/raw"},
            "events": {
                "standard_codes": [1],
                "deviant_codes": [2],
                "behavioral_keep_codes": [1],
                "condition_map": {"A": [1], "B": [1], "C": ["bad"]},
            },
        }
    )

    with pytest.raises(ValueError) as exc_info:
        _validate_config(cfg)

    msg = str(exc_info.value)
    assert "events.condition_map has duplicate code: 1" in msg
    assert "events.condition_map values must be integers" in msg
    assert "events.deviant_codes must be included" in msg


def test_validate_config_rejects_non_mapping_condition_map_and_non_list_windows():
    cfg = _apply_defaults(
        {
            "input": {"mode": "other"},
            "paths": {"bids_root": "/tmp/bids"},
            "events": {
                "standard_codes": [1],
                "deviant_codes": [1],
                "condition_map": ["bad"],
            },
            "metrics": {"erp": {"windows": "bad"}},
        }
    )

    with pytest.raises(ValueError) as exc_info:
        _validate_config(cfg)

    msg = str(exc_info.value)
    assert "input.mode must be one of" in msg
    assert "events.condition_map must be a mapping" in msg
    assert "Standard/deviant code overlap not allowed" in msg
    assert "metrics.erp.windows must be a list" in msg
