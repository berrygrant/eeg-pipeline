# v2.0

## Summary
v2.0 makes the pipeline **BIDS-first** and **task-agnostic**. The default input
is now an existing BIDS EEG dataset and output is written as a BIDS derivatives
dataset; the original lab-folder layout is still supported behind `--legacy`.
The CLI was reorganized from a single module into a focused set of
orchestration modules, and an early Electron "OneClick" GUI prototype was added.

## Highlights
- **BIDS by default.** Input defaults to a BIDS EEG dataset; derivatives are
  written under `derivatives/eeg-pipeline/` with JSON sidecars for provenance.
- **Legacy layout is opt-in** via `--legacy`, with optional legacy→BIDS
  conversion via `--convert_to_bids`.
- **Task-agnostic.** Originally MMN-focused, the pipeline now supports arbitrary
  ERP and time–frequency designs (condition maps, configurable windows).
- **ERP CORE preset** (`--erp-core`) applying Kappenman et al. (2021)-style
  defaults.
- **OneClick GUI prototype** (Electron shell + local Python backend) that reuses
  the existing CLI as its execution engine.
- Optional **GPU acceleration** (MNE CUDA / CuPy) with CPU fallback.

## Changes
- Split the monolithic CLI into `cli`, `cli_parser`, `cli_config`,
  `cli_common`, `cli_pipeline`, `cli_metrics`, `cli_summary`, and `cli_figures`.
- Added BIDS discovery, derivatives creation, and structural validation
  (`bids.py`) plus legacy→BIDS conversion (`inputs.py`).
- Standardized derivative outputs (preprocessed FIF, epochs, aligned events,
  condition evokeds, grand averages, ERP/TFR metrics, QC summary).

## Validation
- Test suite includes a tiny BIDS fixture and validates both the input dataset
  and the produced derivatives tree.

## Known Issues
- GitHub release creation still requires a release object on GitHub for Zenodo
  to archive the new version.
- The OneClick GUI is an early prototype (see README) and is not yet packaged
  for distribution.
