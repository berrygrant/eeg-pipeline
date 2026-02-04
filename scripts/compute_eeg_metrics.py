#!/usr/bin/env python3
"""Opinionated wrapper: compute ERP/TFR metrics from existing epochs."""
from __future__ import annotations

import argparse

from eeg_pipeline.cli import main as pipeline_main


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compute ERP/TFR metrics from existing epochs")
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    args, extras = ap.parse_known_args(argv)

    pipeline_main(["--config", args.config, "--get_metrics", *extras])


if __name__ == "__main__":
    main()
