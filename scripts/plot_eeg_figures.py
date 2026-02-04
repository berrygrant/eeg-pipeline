#!/usr/bin/env python3
"""Opinionated wrapper: generate paper-ready figures from metrics outputs."""
from __future__ import annotations

import argparse

from eeg_pipeline.cli import main as pipeline_main


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate paper-ready figures from metrics outputs")
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    args, extras = ap.parse_known_args(argv)

    pipeline_main(["--config", args.config, "--plot_figures", *extras])


if __name__ == "__main__":
    main()
