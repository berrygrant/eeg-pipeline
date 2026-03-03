import json
from pathlib import Path

import pytest

from eeg_pipeline.config import (
    _normalize_token_map,
    _parse_n_components,
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
