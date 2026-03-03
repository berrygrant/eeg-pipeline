from pathlib import Path

import pandas as pd

from eeg_pipeline.qc import write_qc_summary


def test_write_qc_summary_persists_rows_to_csv(tmp_path: Path):
    out_csv = tmp_path / "qc_summary.csv"
    rows = [
        {"subject": "001", "rejected_epochs": 3},
        {"subject": "002", "rejected_epochs": 1},
    ]

    df = write_qc_summary(rows, out_csv)

    assert out_csv.exists()
    assert list(df["subject"]) == ["001", "002"]

    written = pd.read_csv(out_csv)
    assert list(written["rejected_epochs"]) == [3, 1]
