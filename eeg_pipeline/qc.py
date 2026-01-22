# mmn_pipeline/qc.py
from __future__ import annotations

import pandas as pd


def write_qc_summary(rows: list[dict], out_csv):
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df