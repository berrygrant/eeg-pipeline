from __future__ import annotations

import sys
from pathlib import Path

from . import cli_config as _config
from . import cli_figures as _figures
from . import cli_metrics as _metrics
from . import cli_pipeline as _pipeline
from . import cli_summary as _summary
from .cli_common import (
    _finalize_runtime_paths,
    _pipeline_dataset_root,
    dataset_derivative_path,
)
from .cli_parser import build_arg_parser, build_defaults
from .gpu import capability_report, format_capability_report
from .gpu import configure as configure_gpu


def apply_config(args, defaults=None):
    return _config.apply_config(args, defaults)


def apply_erp_core_preset(args, defaults):
    return _config.apply_erp_core_preset(args, defaults)


def summarize_one_file(args, raw_path: Path):
    return _summary.summarize_one_file(args, raw_path)


def run_legacy_to_bids_conversion(args, defaults=None, cfg=None):
    return _pipeline.run_legacy_to_bids_conversion(args, defaults=defaults, cfg=cfg)


def run_full_pipeline(args, defaults=None, cfg=None):
    return _pipeline.run_full_pipeline(args, defaults=defaults, cfg=cfg)


def run_metrics_only(args):
    return _metrics.run_metrics_only(args)


def run_plot_figures(args):
    return _figures.run_plot_figures(args)


def prepare_output_dirs(out_dir: Path):
    return _figures.prepare_output_dirs(out_dir)


def _subject_from_epochs_path(p: Path) -> str:
    return _metrics._subject_from_epochs_path(p)


def _resolve_figure_time_window(args) -> tuple[float, float]:
    return _metrics._resolve_figure_time_window(args)


def _resolve_figure_freq_band(args) -> tuple[float, float] | None:
    return _metrics._resolve_figure_freq_band(args)


def _build_erp_windows(args):
    return _metrics._build_erp_windows(args)


def _prompt_yes_no(msg: str) -> bool:
    return _figures._prompt_yes_no(msg)


def main(argv=None):
    ap = build_arg_parser()
    defaults = build_defaults(ap)
    args = ap.parse_args(argv)
    stages_requested = bool(args.process_data or args.get_metrics or args.plot_figures)

    if args.summarize_one_file:
        apply_erp_core_preset(args, defaults)
        cfg = apply_config(args, defaults)
        _finalize_runtime_paths(args, cfg)
        summarize_one_file(args, Path(args.summarize_one_file))
        return

    if not stages_requested:
        if bool(getattr(args, "convert_to_bids", False)):
            args.process_data = False
            args.get_metrics = False
            args.plot_figures = False
        else:
            args.process_data = True
            args.get_metrics = True
            args.plot_figures = False

    apply_erp_core_preset(args, defaults)
    cfg = apply_config(args, defaults)
    _finalize_runtime_paths(args, cfg)
    if not stages_requested and bool(getattr(args, "convert_to_bids", False)):
        args.process_data = False
        args.get_metrics = False
        args.plot_figures = False

    if getattr(args, "_erp_core_preset_enabled", False):
        print("[ERP-CORE] preset enabled")
        print(
            "[ERP-CORE] reref={reref} l_freq={l_freq} h_freq={h_freq} "
            "volt_method={volt_method} volt_auto_percentile={volt_auto} "
            "blink_auto_percentile={blink_auto} ica={ica}".format(
                reref=getattr(args, "reref", ""),
                l_freq=getattr(args, "l_freq", ""),
                h_freq=getattr(args, "h_freq", ""),
                volt_method=getattr(args, "volt_method", ""),
                volt_auto=getattr(args, "volt_auto_percentile", ""),
                blink_auto=getattr(args, "blink_auto_percentile", ""),
                ica=getattr(args, "ica", ""),
            )
        )

    gpu_status = configure_gpu(bool(args.use_gpu), device=args.gpu_device)
    if args.use_gpu:
        cap_msg = format_capability_report(capability_report())
        if cap_msg:
            print(cap_msg)
        if gpu_status["enabled"]:
            print(
                "[GPU] enabled (backend="
                f"{gpu_status['backend']}, mne_cuda={gpu_status['mne_cuda']}, "
                f"cupy={gpu_status['cupy']})"
            )
        else:
            print(
                "[WARN] GPU requested but not available; falling back to CPU "
                f"(mne_cuda={gpu_status['mne_cuda']}, cupy={gpu_status['cupy']})"
            )

    if args.plot_figures:
        args.metrics_erp_timeseries = True

    if bool(getattr(args, "convert_to_bids", False)) and not (
        args.process_data or args.get_metrics or args.plot_figures
    ):
        run_legacy_to_bids_conversion(args, defaults=defaults, cfg=cfg)
        return

    if args.process_data:
        args.metrics = 1 if args.get_metrics else 0
        run_full_pipeline(args, defaults=defaults, cfg=cfg)
    elif args.get_metrics:
        run_metrics_only(args)

    if args.plot_figures:
        dataset_root = _pipeline_dataset_root(args)
        erp_parquet = dataset_derivative_path(dataset_root, suffix="timeseries", extension=".parquet", desc="erp")
        tfr_file = dataset_derivative_path(dataset_root, suffix="metrics", extension=".tsv", desc="tfr")

        missing = []
        if not erp_parquet.exists():
            missing.append(str(erp_parquet))
        if not tfr_file.exists():
            missing.append(str(tfr_file))

        if missing:
            print(f"[WARN] Missing figure inputs: {', '.join(missing)}")
            if _prompt_yes_no("Run full pipeline now? [y/N] "):
                args.process_data = True
                args.get_metrics = True
                args.metrics = 1
                run_full_pipeline(args, defaults=defaults, cfg=cfg)
            else:
                print("[WARN] Proceeding with available metrics only.")

        run_plot_figures(args)


if __name__ == "__main__":
    main(sys.argv[1:])
