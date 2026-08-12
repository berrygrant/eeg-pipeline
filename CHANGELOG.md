# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- A config section written as a non-mapping had the user's value silently
  discarded and defaults back-filled over it. `ica: on` — the `mode:` nesting
  omitted — became `ica.mode = "off"`, so the run completed with ICA quietly
  disabled, every subject recording `ica_ran=False`, and no ICA files written.
  Unknown-key detection could not catch it either, since it inspects the config
  only after defaults have rewritten the node into a valid shape. Malformed
  sections are now rejected with the correct nesting shown.
- Duplicate YAML keys are now rejected. YAML keeps only the last occurrence and
  says nothing, so appending a second `ica:` block left a file whose visible text
  still read `mode: "on"` while the run used the later block — evidence and
  behavior disagreeing with nothing to flag it.
- Aggregation keyed recordings on only four of the six BIDS entities that appear
  in derivative filenames, omitting `acq` and `recording`. Two sibling recordings
  differing solely by `acq` therefore shared one key, so excluding the bad sibling
  silently dropped the good one's metrics while its QC row still read OK. QC rows
  now carry `acq`/`recording`, and the key uses all six. QC files written before
  those columns existed still match, and a dataset that uses `acq` fails open
  rather than excluding on a partial key.
- `compute_erp_metrics` returning zero rows wrote a header-only TSV, which passed
  every presence check and was then discarded by the aggregator's empty-frame
  test with nothing logged. The subject vanished from the metrics table while its
  QC row read OK. This case is now reported. (Its sibling
  `compute_erp_timeseries` already emitted an `EMPTY_EPOCHS` sentinel row for the
  same situation.)
- Config values that collide with YAML 1.1 booleans were mishandled in two
  opposite directions, and the two compounded into a trap.
  - `ica.mode: on` and `ica.mode: off` — two of the three documented modes — were
    rejected, because PyYAML resolves bare `on`/`off` to `True`/`False` and the
    loader stringified those to `"true"`/`"false"`. The error then reported that
    the value "must be one of: off | auto | on", listing the very word the user
    had written. Quoting was the only way through and nothing said so. The same
    applied to `preprocess.reref: no`, an accepted alias.
  - Boolean options were decided by Python truthiness, so a quoted
    `compute.use_gpu: "off"` **enabled** the GPU — `bool("off")` is `True`. A user
    who hit the first problem and learned to quote values would land directly on
    this one. All eight schema booleans are now parsed by word, and an
    unrecognized value raises instead of defaulting to `True`.


## [2.2.0] - 2026-08-12

### Added
- **HPC/SLURM mode.** Subjects can now be processed by independent jobs and
  combined afterwards. `--aggregate_only` rebuilds the dataset-level QC/metrics
  tables and grand averages from per-subject derivatives on disk; `run_full_pipeline`
  calls the same code as its tail, so serial and array-parallel runs cannot drift.
  Aggregation overwrites its outputs, so a gather job is safe to re-run after a
  partial array. `--skip_aggregate` suppresses the per-run rebuild and is required
  for concurrent per-subject jobs, which would otherwise each rebuild the shared
  dataset-level tables and race one another. Ships `hpc/slurm_array.sbatch`, `hpc/slurm_gather.sbatch`, and
  `hpc/submit.sh`, plus a README "HPC / SLURM" section.
- `compute.n_jobs` / `--n_jobs` threads MNE's channel-parallel `n_jobs` into
  filtering, notch, the ICA pre-fit filter, and `compute_tfr`. `ICA.fit` takes no
  `n_jobs` and remains serial. Invalid values (0, negatives other than -1,
  non-integers) are rejected at config load.
- Per-subject QC rows are now written next to a recording's other derivatives,
  including when the recording raised — previously QC existed only in memory.

### Fixed
- `--get_metrics` with subject filters destroyed other subjects' rows in the
  dataset-level tables. `run_metrics_only` concatenated only the frames that run
  computed and wrote them to the dataset-level paths, so once filtering existed a
  per-subject rerun overwrote the combined table with a single subject. It now
  rebuilds those tables from the per-subject files on disk (the same aggregation
  `run_full_pipeline` uses) and honors `--skip_aggregate`.
- `hpc/submit.sh` sized the array by non-blank line count while
  `hpc/slurm_array.sbatch` selected subjects by physical line number. One blank
  line desynchronized them: tasks landed on empty lines, exited non-zero, and the
  `afterok`-chained gather job never ran. Both now index non-blank lines.
- A recording that failed before any stage recorded an outcome (raw loading,
  filtering) left no QC row at all, so it vanished from the QC summary — and,
  because aggregation excludes recordings by QC status, could not be excluded
  from the dataset tables either. Such failures now write an `ERROR` QC row.
- Dataset-level outputs from a previous gather survived one that had no inputs to
  rebuild them from (all subjects excluded, a condition no longer present, or
  unreadable evokeds), so `--plot_figures` consumed them as current. Aggregation
  now removes dataset-level outputs it cannot rebuild. Per-subject data is never
  touched.
- Aggregation folded stale per-subject files from earlier runs back into the
  dataset-level outputs. Re-running with a stricter setting (e.g. a tighter
  `--max_reject_rate`) rewrote a recording's QC row to a skip status but left its
  previous metrics and evokeds on disk, so the dataset reported a subject excluded
  while still averaging it into the combined tables and grand averages.
  Aggregation now treats the QC status as authoritative and excludes recordings
  the latest run did not process successfully.
- Grand averages grouped conditions by raw label while the output path lower-cases
  them, so "Standard" (sidecar) and "standard" (filename fallback) produced two
  groups writing to the same path — one silently overwriting the other, each built
  from part of the cohort. Conditions are now grouped case-insensitively.
- Aggregation read zero-padded BIDS entity labels back as integers (`run` "01" → 1),
  making the combined tables disagree with the filenames and the per-subject tables.
  Entity columns now round-trip as strings.
- `gpu.filter_n_jobs()` treated "init_cuda() did not raise" as CUDA capability.
  MNE's `init_cuda` no-ops silently when `MNE_USE_CUDA` is not "true" (the
  default) or cupy is missing, so `--use_gpu` on an unconfigured machine routed
  filtering to CUDA; MNE then coerced `n_jobs` to 1 before falling back to the
  CPU, discarding the requested workers and running slower than a plain CPU run.
  Capability is now read from MNE's own flag, failing safe to the CPU.
- An explicit `--n_jobs 1` could not override a larger `compute.n_jobs` in the
  config, because the argparse default was also 1 and provided-flag detection
  compares against it. This mattered for `hpc/slurm_array.sbatch`, which passes
  exactly 1 when `SLURM_CPUS_PER_TASK` is unset — the config value would have
  silently oversubscribed the allocation. The default is now a sentinel.
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
