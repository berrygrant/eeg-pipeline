# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.0] - 2026-07-20

### Fixed
- Infinite recursion on `--legacy --convert_to_bids --process_data` caused by
  the CLI rebinding `cli_pipeline.run_legacy_to_bids_conversion` to a
  self-calling wrapper.
- Config validation now rejects unknown/misspelled keys instead of silently
  falling back to defaults.
- `scripts/import_manual_rejection_sets.py --run_metrics` invoked the CLI with
  an `--out_dir` flag it no longer accepts; it now drives the flat-directory
  analysis runner with the required `--do_erp`/`--do_tfr` selection flags,
  matching the tree it produces.
- Config validation recognizes `paths.out_dir` (read by the manual-rejection
  utility) so shared configs are not rejected as containing an unknown key.
- Packaging omitted `eeg_pipeline.oneclick` and treated `eeg_pipeline.viz` as a
  namespace directory; both are now real, discoverable packages.

### Changed
- Coverage is measured over the whole package again (removed the omit list that
  hid the CLI/orchestration layer, ~44% of LOC); codecov patch gate re-enabled.
- Package version is now sourced dynamically from `eeg_pipeline.__version__`.
- `requires-python` raised to `>=3.10` (matches the documented/tested floor);
  ruff target set to py310.
- Core dependencies carry lower bounds; `matplotlib`/`seaborn` documented as the
  optional `viz` extra; added an `ica` extra for `python-picard`.

### Added
- CI quality gates: ruff lint, a non-blocking mypy job, a Python 3.10/3.11/3.12
  test matrix, and a package-install + console-entry-point smoke check.
- `py.typed` marker (PEP 561), `MANIFEST.in`, and a pre-commit config.
- Documentation of the intentionally-inert TFR baseline keys in `config.yaml`.

### Deprecated
- (Planned) The flat-directory metrics runners (`run_analysis.py`,
  `run_metrics.py`) will be consolidated behind the config-driven BIDS engine.

## [2.0.0] - 2026
- BIDS-first, task-agnostic pipeline; CLI split into modules; OneClick GUI
  prototype; ERP CORE preset; optional GPU acceleration. See
  `RELEASE_NOTES_v2.0.md`.

## [1.1] - 2026
- Manual-rejection import honors EEGLAB trial-rejection flags; configurable
  rejection modes; richer import summaries. See `RELEASE_NOTES_v1.1.md`.

## [1.0]
- GPU/CuPy support, EEGLAB `.set` ingest, windowed artifact rejection, ERP CORE
  preset, post-hoc metrics runner, paper-ready figures. See
  `RELEASE_NOTES_v1.0.md`.
