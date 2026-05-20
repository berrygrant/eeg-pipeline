"""Compatibility wrapper for the package analysis runner."""

from __future__ import annotations

from eeg_pipeline.analysis_runner import (
    _close_figures,
    _iter_figures,
    _label_missing_condition,
    _maybe_make_figures,
    _resolve_erp_windows,
    _save_figures,
    _split_rows_with_missing_condition,
    _subject_from_filename,
    _unlink_if_exists,
    build_arg_parser,
    main,
)

__all__ = [
    "_close_figures",
    "_iter_figures",
    "_label_missing_condition",
    "_maybe_make_figures",
    "_resolve_erp_windows",
    "_save_figures",
    "_split_rows_with_missing_condition",
    "_subject_from_filename",
    "_unlink_if_exists",
    "build_arg_parser",
    "main",
]


if __name__ == "__main__":
    main()
