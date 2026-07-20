from __future__ import annotations

import pandas as pd


def write_qc_summary(rows: list[dict], out_csv):
    df = pd.DataFrame(rows)
    sep = "\t" if str(out_csv).endswith(".tsv") else ","
    df.to_csv(out_csv, sep=sep, index=False)
    return df
