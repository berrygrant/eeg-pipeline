#!/usr/bin/env python3
"""Manual EEG review launcher (raw/epochs) with sidecar export."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct execution from the repository root without package install.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eeg_pipeline.manual_review import default_sidecar_path, review_file


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Open interactive manual review and save rejection sidecar JSON."
    )
    ap.add_argument("--input", required=True, help="Path to input file (.vhdr, .set, .fif).")
    ap.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "raw", "epochs"],
        help="Review mode. 'auto' infers from file name/extension.",
    )
    ap.add_argument(
        "--sidecar",
        default=None,
        help="Output sidecar path (default: <input>.<ext>.manual_reject.json).",
    )
    ap.add_argument(
        "--save_cleaned",
        default=None,
        help="Optional output path for cleaned raw/epochs .fif file.",
    )
    ap.add_argument(
        "--no_block",
        action="store_true",
        help="Open browser non-blocking (advanced/debug use).",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    input_path = Path(args.input)
    sidecar = Path(args.sidecar) if args.sidecar else default_sidecar_path(input_path)
    save_cleaned = Path(args.save_cleaned) if args.save_cleaned else None

    result = review_file(
        input_path=input_path,
        mode=args.mode,
        sidecar_path=sidecar,
        save_cleaned_path=save_cleaned,
        block=not args.no_block,
    )

    print(
        "[OK] Manual review complete:",
        f"mode={result.mode}",
        f"sidecar={result.sidecar_path}",
        f"bad_channels={len(result.bad_channels)}",
        f"annotations={result.n_annotations}",
        f"dropped_epochs={result.n_dropped_epochs}",
    )
    if result.cleaned_output_path is not None:
        print(f"[OK] Saved cleaned file -> {result.cleaned_output_path}")


if __name__ == "__main__":
    main()
