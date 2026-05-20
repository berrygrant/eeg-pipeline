import pytest

from eeg_pipeline.schema import (
    decode_eventcode_v1,
    derive_metadata_from_condition_map,
    derive_metadata_v1,
    parse_token_map,
)


def test_parse_token_map_supports_defaults_bare_and_keyed_inputs():
    assert parse_token_map(None) == {"token1": "token1", "token2": "token2"}
    assert parse_token_map(["EH", "IH"]) == {"token1": "EH", "token2": "IH"}
    assert parse_token_map(["Token1=EH", "t2=IH"]) == {"token1": "EH", "token2": "IH"}


def test_parse_token_map_rejects_unknown_and_incomplete_inputs():
    with pytest.raises(ValueError, match="Unknown --token_map key"):
        parse_token_map(["token3=EH", "IH"])

    with pytest.raises(ValueError, match="Incomplete --token_map"):
        parse_token_map(["EH"])


def test_decode_eventcode_v1_splits_hundreds_tens_and_ones():
    assert decode_eventcode_v1(210) == {"A": 2, "B": 1, "C": 0}


def test_derive_metadata_v1_builds_expected_labels():
    df = derive_metadata_v1([110, 111, 210], token_map={"token1": "EH", "token2": "IH"})

    assert list(df["is_standard"]) == [True, False, True]
    assert list(df["is_deviant"]) == [False, True, False]
    assert list(df["vowel_variant"]) == ["full", "full", "full"]
    assert list(df["standard_token_role"]) == ["token1", "token1", "token2"]
    assert list(df["deviant_token_role"]) == ["token2", "token2", "token1"]
    assert list(df["trial_token"]) == ["EH", "IH", "IH"]


def test_derive_metadata_v1_uses_na_sentinel_for_practice_blocks():
    df = derive_metadata_v1([110, 911])

    assert list(df["is_practice"]) == [False, True]
    assert list(df["is_main"]) == [True, False]
    assert list(df["trial_token"]) == ["token1", "NA"]


def test_derive_metadata_from_condition_map_parses_named_and_unknown_conditions():
    df = derive_metadata_from_condition_map(
        [1, 2, 3, 999],
        {
            "ntDontcount_lab": 1,
            "t_count_campus": 2,
            "plain": 3,
        },
    )

    assert list(df["condition"]) == [
        "ntDontcount_lab",
        "t_count_campus",
        "plain",
        "UNKNOWN",
    ]
    assert list(df["stimulus"]) == ["nt", "t", "plain", "UNKNOWN"]
    assert list(df["task"]) == ["Dontcount", "count", "NA", "NA"]
    assert list(df["environment"]) == ["lab", "campus", "NA", "NA"]
    assert list(df["is_standard"]) == [True, False, False, False]
    assert list(df["is_deviant"]) == [False, True, False, False]
