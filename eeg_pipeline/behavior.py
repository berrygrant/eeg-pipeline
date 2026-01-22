# mmn_pipeline/behavior.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def read_eventcodes_from_subject_csv(subject_csv: Path) -> np.ndarray:
    df = pd.read_csv(subject_csv)
    if "EventCode" not in df.columns:
        raise ValueError(
            f"'EventCode' column not found in {subject_csv}. "
            f"Found columns: {list(df.columns)}"
        )
    codes = df["EventCode"].to_numpy()
    if not np.issubdtype(codes.dtype, np.integer):
        codes = codes.astype(int)
    return codes


def filter_codes(codes: np.ndarray, keep_codes: list[int] | None) -> np.ndarray:
    if keep_codes is None or len(keep_codes) == 0:
        return codes
    keep = np.isin(codes, np.asarray(keep_codes, dtype=int))
    return codes[keep]