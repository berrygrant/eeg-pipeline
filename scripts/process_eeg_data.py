#!/usr/bin/env python3
"""Opinionated wrapper: process raw EEG into epochs/evokeds/QC."""
from __future__ import annotations

import argparse

from eeg_pipeline.cli import main as pipeline_main


def main(argv=None):
    ap = argparse.ArgumentParser(description="Process raw EEG data (preprocessing + QC)")
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    args, extras = ap.parse_known_args(argv)

    pipeline_main(["--config", args.config, "--process_data", *extras])


if __name__ == "__main__":
    main()
