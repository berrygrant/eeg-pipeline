import json
from pathlib import Path
import sys
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


def test_load_config_json_applies_defaults_and_normalizes_types(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "paths": {
                    "raw_dir": "/tmp/raw",
                    "subject_csv_dir": "/tmp/subject_csv",
                    "out_dir": "/tmp/out",
                },
                "events": {
                    "standard_codes": ["110"],
                    "deviant_codes": ["111"],
                    "behavioral_keep_codes": ["110", "111"],
                },
                "ica": {"n_components": "20"},
                "labels": {"token_map": ["Token1=EH", "Token2=IH"]},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg["paths"]["raw_dir"] == Path("/tmp/raw")
    assert cfg["paths"]["subject_csv_dir"] == Path("/tmp/subject_csv")
    assert cfg["paths"]["out_dir"] == Path("/tmp/out")
    assert cfg["channels"]["blink_proxy_chs"] == ["Fp1"]
    assert cfg["preprocess"]["notch_hz"] == [60.0]
    assert cfg["events"]["standard_codes"] == [110]
    assert cfg["events"]["deviant_codes"] == [111]
    assert cfg["ica"]["n_components"] == 20
    assert cfg["labels"]["token_map"] == {"token1": "EH", "token2": "IH"}
    assert cfg["metrics"]["erp"]["enabled"] is True


def test_load_config_rejects_invalid_overlapping_standard_and_deviant_codes(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "paths": {
                    "raw_dir": "/tmp/raw",
                    "subject_csv_dir": "/tmp/subject_csv",
                    "out_dir": "/tmp/out",
                },
                "events": {
                    "standard_codes": [110],
                    "deviant_codes": [110],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Standard/deviant code overlap not allowed"):
        load_config(cfg_path)


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
    assert _normalize_token_map("") is None


def _valid_min_cfg():
    return {
        "paths": {
            "raw_dir": "/tmp/raw",
            "subject_csv_dir": "/tmp/subject_csv",
            "out_dir": "/tmp/out",
        },
        "events": {
            "standard_codes": [1],
            "deviant_codes": [2],
            "behavioral_keep_codes": [1, 2],
        },
    }


def test_load_config_rejects_missing_file_and_non_mapping_top_level(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(tmp_path / "missing.json")

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="Top-level config must be a mapping"):
        load_config(cfg_path)


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
    cfg["compute"]["use_gpu"] = 1
    cfg["compute"]["gpu_device"] = ""

    normalized = _normalize_config(cfg)

    assert normalized["paths"]["raw_dir"] == Path("/tmp/raw")
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
    assert normalized["compute"]["use_gpu"] is True
    assert normalized["compute"]["gpu_device"] is None


def test_scalar_list_helpers_and_condition_map_normalization_cover_edge_cases():
    assert _as_int_list(None) == []
    assert _as_int_list("5") == [5]
    assert _as_float_list(None) == []
    assert _as_float_list("1.25") == [1.25]
    assert _normalize_condition_map(None) is None
    assert _normalize_condition_map({"A": "1", "B": [2, 3]}) == {"A": [1], "B": [2, 3]}

    with pytest.raises(ValueError, match="mapping of name"):
        _normalize_condition_map(["bad"])


def test_normalize_token_map_covers_keyed_list_and_single_string_paths():
    assert _normalize_token_map(["skip", "Other=ZZ", "Token2=IH"]) == {"token2": "IH"}
    assert _normalize_token_map("EH IH") == {"token1": "EH", "token2": "IH"}
    assert _normalize_token_map("Token1=EH") is None


def test_validate_config_collects_multiple_readable_errors():
    cfg = _apply_defaults(_valid_min_cfg())
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
    assert "epoching.baseline must be a 2-item list" in msg
    assert "ica.mode must be one of" in msg
    assert "preprocess.reref must be one of" in msg
    assert "artifacts.voltage.method must be one of" in msg
    assert "artifacts.blink.auto_percentile must be in (0, 100]" in msg
    assert "artifacts.voltage.auto_percentile must be a number" in msg
    assert "events.standard_codes / deviant_codes must be integers" in msg
    assert "events.condition_map['A'] must map to a single code" in msg
    assert "metrics.erp.windows[0] must include name, tmin, tmax" in msg
    assert "events.behavioral_keep_codes must be integers" in msg


def test_validate_config_flags_missing_required_fields_and_subset_rules():
    cfg = _apply_defaults({})
    cfg["events"]["condition_map"] = "bad"
    cfg["metrics"]["erp"]["windows"] = "bad"

    with pytest.raises(ValueError) as exc_info:
        _validate_config(cfg)

    msg = str(exc_info.value)
    assert "Missing required field: 'paths.raw_dir'" in msg
    assert "Missing required field: 'paths.subject_csv_dir'" in msg
    assert "Missing required field: 'paths.out_dir'" in msg
    assert "events.standard_codes and events.deviant_codes must both be non-empty" in msg
    assert "events.condition_map must be a mapping of name -> code(s)" in msg
    assert "metrics.erp.windows must be a list" in msg


def test_validate_config_detects_duplicate_condition_codes_and_keep_code_subsets():
    cfg = _apply_defaults(_valid_min_cfg())
    cfg["events"]["condition_map"] = {"Std": 1, "Dup": 1}
    cfg["events"]["behavioral_keep_codes"] = [3]

    with pytest.raises(ValueError) as exc_info:
        _validate_config(cfg)

    msg = str(exc_info.value)
    assert "events.condition_map has duplicate code: 1" in msg
    assert "events.standard_codes must be included in events.behavioral_keep_codes" in msg
    assert "events.deviant_codes must be included in events.behavioral_keep_codes" in msg


def test_load_config_treats_json_null_as_empty_config(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text("null", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required field: 'paths.raw_dir'"):
        load_config(cfg_path)


def test_validate_config_covers_remaining_error_branches():
    cfg = _apply_defaults(_valid_min_cfg())
    cfg["events"]["condition_map"] = {"Bad": object()}
    cfg["metrics"]["erp"]["windows"] = ["bad"]
    cfg["artifacts"]["blink"]["auto_percentile"] = "oops"
    cfg["artifacts"]["voltage"]["auto_percentile"] = "101"

    with pytest.raises(ValueError) as exc_info:
        _validate_config(cfg)

    msg = str(exc_info.value)
    assert "artifacts.blink.auto_percentile must be a number" in msg
    assert "artifacts.voltage.auto_percentile must be in (0, 100]" in msg
    assert "events.condition_map values must be integers" in msg
    assert "metrics.erp.windows[0] must be a mapping" in msg


def test_normalize_config_handles_list_conditions_and_parse_n_components_error_path():
    cfg = _apply_defaults(_valid_min_cfg())
    cfg["metrics"]["erp"]["conditions"] = ["Oddball", "Rare"]

    normalized = _normalize_config(cfg)

    assert normalized["metrics"]["erp"]["conditions"] == ["Oddball", "Rare"]

    with pytest.raises(ValueError):
        _parse_n_components("abc")
