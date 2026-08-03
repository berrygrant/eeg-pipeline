# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **HPC/SLURM mode.** Subjects can now be processed by independent jobs and
  combined afterwards. `--aggregate_only` rebuilds the dataset-level QC/metrics
  tables and grand averages from per-subject derivatives on disk; `run_full_pipeline`
  calls the same code as its tail, so serial and array-parallel runs cannot drift.
  Aggregation overwrites its outputs, so a gather job is safe to re-run after a
  partial array. Ships `hpc/slurm_array.sbatch`, `hpc/slurm_gather.sbatch`, and
  `hpc/submit.sh`, plus a README "HPC / SLURM" section.
- `compute.n_jobs` / `--n_jobs` threads MNE's channel-parallel `n_jobs` into
  filtering, notch, the ICA pre-fit filter, and `compute_tfr`. `ICA.fit` takes no
  `n_jobs` and remains serial. Invalid values (0, negatives other than -1,
  non-integers) are rejected at config load.
- Per-subject QC rows are now written next to a recording's other derivatives,
  including when the recording raised — previously QC existed only in memory.

### Fixed
- GPU acceleration was inert: `configure()` initialized MNE CUDA, but MNE only
  routes filtering through CUDA when `n_jobs="cuda"` and the package never passed
  `n_jobs` at all, so `use_gpu` had no effect on filtering. New
  `gpu.filter_n_jobs()` routes correctly and falls back to CPU workers otherwise.
  (Scope is unchanged: MNE CUDA covers FFT filtering/resampling, not ICA or TFR.)
- `--get_metrics` ignored `--subjects`/`--sessions`/`--tasks`/`--runs`: the metrics
  stage globbed every `*_epo.fif` in the derivatives tree regardless of the
  requested filters. A per-subject invocation therefore recomputed every subject.
  Added `bids.filter_derivative_paths`, which applies the same entity-filter
  semantics as `discover_bids_eeg_recordings` (bare or `sub-` prefixed) to
  derivative filenames, and a distinct error when filters match nothing.

### Added
- Per-stage wall-clock timing (`cli_common.StageTimings`) recorded into each QC
  row as `t_<stage>_s` columns for `preprocess`, `ica`, `epoching`, `metrics`, and
  `io`. Timings are merged onto every row a recording produces — including
  early-skip rows and rows from a recording that raised — so one real run reports
  where time actually goes instead of leaving it to be estimated.

### Deprecated
- The flat-directory metrics runners now emit a `DeprecationWarning` at runtime.
  This covers the `eeg-run-analysis` console script, the root `run_analysis.py`
  and `run_metrics.py` wrappers, and `python -m eeg_pipeline.analysis_runner` —
  all of which funnel through `analysis_runner.main`. They continue to work
  unchanged for now; prefer the config-driven, BIDS-based
  `python -m eeg_pipeline.cli --get_metrics`, which routes through the same
  `eeg_pipeline.metrics` engine (identical ERP/TFR numerics and one baseline
  policy). The flat-directory runners will be removed in a future release.

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
