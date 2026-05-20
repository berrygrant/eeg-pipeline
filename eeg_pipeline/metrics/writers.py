from __future__ import annotations

from pathlib import Path

import pandas as pd


def append_csv(df: pd.DataFrame, path: Path) -> None:
    """Append a DataFrame to a CSV, writing the header only once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


class ParquetRowGroupWriter:
    """Write DataFrames as a Parquet dataset without retaining frames in memory."""

    def __init__(self) -> None:
        self._counts: dict[Path, int] = {}

    def write(self, df: pd.DataFrame, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        idx = self._counts.get(path, 0)
        self._counts[path] = idx + 1
        df.to_parquet(path / f"part-{idx:05d}.parquet", index=False)

    def close(self) -> None:
        self._counts.clear()

    def __enter__(self) -> ParquetRowGroupWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def reset_combined_metric_outputs(metrics_dir: Path) -> None:
    import shutil

    for name in ("erp_metrics_all.csv", "erp_timeseries_all.parquet", "tfr_metrics_all.csv"):
        path = metrics_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
