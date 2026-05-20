# ruff: noqa: F403,F405
from __future__ import annotations

from . import cli_common as _common
from .cli_common import *  # noqa: F403
from .cli_metrics import _resolve_figure_freq_band, _resolve_figure_time_window

globals().update({name: value for name, value in vars(_common).items() if not name.startswith("__") or name == "__version__"})


def _prompt_yes_no(msg: str) -> bool:
    if not sys.stdin.isatty():
        return False
    resp = input(msg).strip().lower()
    return resp in {"y", "yes"}


def run_plot_figures(args):
    from eeg_pipeline.viz import paper_figures

    _finalize_runtime_paths(args)
    dataset_root = _pipeline_dataset_root(args)
    _dataset_metrics_dir(dataset_root)
    fig_dir = Path(args.figures_out_dir) if args.figures_out_dir else dataset_root / "figures"

    erp_parquet = dataset_derivative_path(dataset_root, suffix="timeseries", extension=".parquet", desc="erp")
    tfr_file = dataset_derivative_path(dataset_root, suffix="metrics", extension=".tsv", desc="tfr")

    erp_exists = erp_parquet.exists()
    tfr_exists = tfr_file.exists()

    if not erp_exists and not tfr_exists:
        raise FileNotFoundError(
            f"No metrics found for plotting. Expected {erp_parquet} and/or {tfr_file}."
        )

    time_window = _resolve_figure_time_window(args)
    freq_band = _resolve_figure_freq_band(args) if tfr_exists else None

    argv = [
        "--out_dir", str(fig_dir),
        "--time_window", str(time_window[0]), str(time_window[1]),
    ]
    if erp_exists:
        argv += ["--erp_parquet", str(erp_parquet)]
    if tfr_exists and freq_band is not None:
        argv += [
            "--tfr_file", str(tfr_file),
            "--freq_band", str(freq_band[0]), str(freq_band[1]),
        ]
    if args.figure_diff_heatmap:
        argv.append("--diff_heatmap")
    if args.figure_channels:
        argv += ["--channels", *args.figure_channels]

    paper_figures.main(argv)

def prepare_output_dirs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_derivatives_dataset(out_dir, pipeline_version=__version__)
