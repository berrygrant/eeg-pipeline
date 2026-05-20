import pandas as pd

from eeg_pipeline.metrics.writers import (
    ParquetRowGroupWriter,
    append_csv,
    reset_combined_metric_outputs,
)


def test_append_csv_writes_header_once(tmp_path):
    path = tmp_path / "metrics.csv"

    append_csv(pd.DataFrame({"subject": ["s1"], "value": [1]}), path)
    append_csv(pd.DataFrame({"subject": ["s2"], "value": [2]}), path)

    assert path.read_text(encoding="utf-8").splitlines() == [
        "subject,value",
        "s1,1",
        "s2,2",
    ]


def test_parquet_writer_streams_readable_dataset(tmp_path):
    path = tmp_path / "erp_timeseries_all.parquet"
    writer = ParquetRowGroupWriter()

    writer.write(pd.DataFrame({"subject": ["s1"], "value": [1.0]}), path)
    writer.write(pd.DataFrame({"subject": ["s2"], "value": [2.0]}), path)
    writer.close()

    df = pd.read_parquet(path).sort_values("subject").reset_index(drop=True)
    assert df.to_dict(orient="list") == {"subject": ["s1", "s2"], "value": [1.0, 2.0]}


def test_reset_combined_metric_outputs_removes_file_and_dataset(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "erp_metrics_all.csv").write_text("x\n", encoding="utf-8")
    dataset = metrics_dir / "erp_timeseries_all.parquet"
    dataset.mkdir()
    (dataset / "part-00000.parquet").write_text("not real parquet", encoding="utf-8")

    reset_combined_metric_outputs(metrics_dir)

    assert not (metrics_dir / "erp_metrics_all.csv").exists()
    assert not dataset.exists()
