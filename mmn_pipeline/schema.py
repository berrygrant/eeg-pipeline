# mmn_pipeline/schema.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class SchemaV1Config:
    """
    ABC schema:
      A: loop type / block type
      B: buffer flag (0 buffer, 1 true)
      C: condition (0 standard, 1 deviant)

    Token roles:
      Token1/Token2 are abstract identities supplied at runtime via --token_map.
      Which token is standard in a block is determined by A.
    """
    # For block types where Token1 is the *standard* token
    token1_standard_A: tuple[int, ...] = (1, 3)
    # For block types where Token2 is the *standard* token
    token2_standard_A: tuple[int, ...] = (2, 4)

    # Reduced/full mapping by A (customize as needed)
    full_A: tuple[int, ...] = (1, 2, 9)
    reduced_A: tuple[int, ...] = (3, 4, 8)

    practice_A: tuple[int, ...] = (8, 9)


def parse_token_map(args: Optional[Iterable[str]]) -> dict[str, str]:
    """
    Accepts:
      --token_map EH IH
      --token_map Token1=EH Token2=IH
      --token_map EH Token2=IH
    Returns:
      {"token1": "EH", "token2": "IH"}
    Defaults to:
      {"token1": "token1", "token2": "token2"}
    """
    default = {"token1": "token1", "token2": "token2"}
    if not args:
        return default

    token1 = None
    token2 = None
    bare: list[str] = []

    for item in args:
        if item is None:
            continue
        s = str(item).strip()
        if not s:
            continue
        if "=" in s:
            k, v = s.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if k in ("token1", "t1"):
                token1 = v
            elif k in ("token2", "t2"):
                token2 = v
            else:
                raise ValueError(f"Unknown --token_map key '{k}'. Use Token1/Token2 (or T1/T2).")
        else:
            bare.append(s)

    if token1 is None and len(bare) >= 1:
        token1 = bare[0]
    if token2 is None and len(bare) >= 2:
        token2 = bare[1]

    if len(bare) > 2:
        raise ValueError(
            f"Too many bare token labels in --token_map: {bare}. "
            "Provide exactly two, or use Token1=/Token2=."
        )

    if token1 is None or token2 is None:
        raise ValueError(
            "Incomplete --token_map. Provide either two labels (e.g., --token_map EH IH) "
            "or explicit pairs (e.g., --token_map Token1=EH Token2=IH)."
        )

    return {"token1": token1, "token2": token2}


def decode_eventcode_v1(code: int) -> dict[str, int]:
    A = int(code) // 100
    B = (int(code) // 10) % 10
    C = int(code) % 10
    return {"A": A, "B": B, "C": C}


def derive_metadata_v1(
    codes,
    token_map: dict[str, str] | None = None,
    cfg: SchemaV1Config | None = None,
):
    """
    Return a pandas DataFrame with decoded fields + derived labels.
    Keep schema logic isolated here for easy future updates.
    """
    import numpy as np
    import pandas as pd

    if cfg is None:
        cfg = SchemaV1Config()
    if token_map is None:
        token_map = {"token1": "token1", "token2": "token2"}

    codes_arr = np.asarray(codes, dtype=int)
    A = codes_arr // 100
    B = (codes_arr // 10) % 10
    C = codes_arr % 10

    df = pd.DataFrame({"code": codes_arr, "A": A, "B": B, "C": C})
    df["is_true"] = df["B"] == 1
    df["is_buffer"] = df["B"] == 0
    df["is_standard"] = df["C"] == 0
    df["is_deviant"] = df["C"] == 1

    df["is_practice"] = df["A"].isin(cfg.practice_A)
    df["is_main"] = ~df["is_practice"]

    df["vowel_variant"] = "other"
    df.loc[df["A"].isin(cfg.full_A), "vowel_variant"] = "full"
    df.loc[df["A"].isin(cfg.reduced_A), "vowel_variant"] = "reduced"

    # Which token is standard in this block
    df["standard_token_role"] = "NA"
    df.loc[df["A"].isin(cfg.token1_standard_A), "standard_token_role"] = "token1"
    df.loc[df["A"].isin(cfg.token2_standard_A), "standard_token_role"] = "token2"

    # Deviant token role (two-token design)
    df["deviant_token_role"] = (
        df["standard_token_role"]
        .map({"token1": "token2", "token2": "token1"})
        .fillna("NA")
    )

    # Trial token role depends on standard vs deviant
    is_std = df["is_standard"].to_numpy()
    std_role = df["standard_token_role"].to_numpy()
    dev_role = df["deviant_token_role"].to_numpy()

    df["trial_token_role"] = "NA"
    df.loc[is_std, "trial_token_role"] = std_role[is_std]
    df.loc[~is_std, "trial_token_role"] = dev_role[~is_std]

    # Human-readable labels
    df["standard_token"] = df["standard_token_role"].map(token_map).fillna(df["standard_token_role"])
    df["deviant_token"] = df["deviant_token_role"].map(token_map).fillna(df["deviant_token_role"])
    df["trial_token"] = df["trial_token_role"].map(token_map).fillna(df["trial_token_role"])

    return df