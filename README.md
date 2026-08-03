# eeg-pipeline

A modular, config-driven EEG preprocessing and analysis pipeline built on **MNE-Python**.  
Originally developed for MMN paradigms, the pipeline is now **task-agnostic** and supports
ERP and time–frequency analyses across arbitrary experimental designs.

This project is designed for **research-grade EEG workflows** with an emphasis on:

- Reproducibility
- Auditability (QC summaries, ICA diagnostics)
- MATLAB → MNE conceptual continuity
- Scalable batch processing

Current Release: v2.1.0 | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18704504.svg)](https://doi.org/10.5281/zenodo.18704504)
 | [![codecov](https://codecov.io/gh/berrygrant/eeg-pipeline/graph/badge.svg?token=YFC9JPJUL3)](https://codecov.io/gh/berrygrant/eeg-pipeline)

Questions? Try the [ChatGPT eeg-pipeline Assistant](https://chatgpt.com/g/g-69985fa61c3881918c1621403999cf9d-eeg-pipeline-assistant)

---

## Key Features

### Core preprocessing
- BrainVision (.vhdr / .vmrk) and EEGLAB (.set) input
- Standard montages (e.g., `standard_1020`)
- Configurable re-reference (`average` or `none`)
- Band-pass and notch filtering
- Automatic handling of missing `.vmrk` or behavioral files (skip / warn / fail)

### Event alignment
- Alignment of EEG markers to behavioral event codes
- Gap-based heuristics for boundary marker removal
- Automatic trimming when EEG markers exceed behavioral codes
- Explicit support for **standard vs. deviant** contrasts

### Epoching & artifact rejection
- Configurable epoch windows and baselines
- Blink detection using:
  - True EOG channels (if present)
  - Proxy EEG channels (e.g., Fp1) if EOG is unavailable
- Simple voltage-based artifact rejection
- Transparent reporting of rejected epochs

### ICA (optional)
- Modes:
  - `off` – no ICA
  - `auto` – run ICA only if blink rate exceeds threshold
  - `on` – always run ICA
- Multiple ICA solvers (`fastica`, `picard`, `infomax`)
- Automatic component exclusion using EOG / proxy correlations
- ICA diagnostics and recommendations saved to QC output
- ICA objects optionally saved for reuse and auditing

### ERP outputs
- Condition-wise evoked responses (Standard / Deviant)
- Grand averages across subjects
- ERP window definitions via config (e.g., MMN, N1, P3a, P3b)
- ERP time‑series extraction to Parquet (per subject + combined)

### Time–frequency analysis (optional)
- Evoked + total TFR computation (multitaper or Morlet)
- Derived induced power (total - evoked) and inter-trial coherence (ITC)
- Configurable frequency ranges and time windows
- TF-domain baselines are intentionally **not** applied (ERPLAB‑equivalent); ITC is kept raw
- Fully compatible with MNE `AverageTFR` objects

### Quality control
- Per-subject QC rows written to `qc_summary.csv`
- Includes:
  - Event counts
  - Epoch rejection rates
  - Blink metrics
- ICA decisions and exclusions
- Designed to support downstream statistical screening

### Visualization (paper-ready figures)
- ERP grand averages (all electrodes or per-electrode)
- TFR time-series (evoked power, ITC) in a time/frequency window
- TFR heatmaps (side-by-side + optional deviant–standard difference)
- Half‑violin plots of evoked/induced power and ITC
- Power plots use **log10 transform after averaging** (ERPLAB‑style)

---

## Repository structure

```
eeg_pipeline/
├── cli.py                # Main entry point (python -m eeg_pipeline.cli)
├── cli_parser.py         # Argument parser + argparse defaults
├── cli_config.py         # Config ↔ CLI-arg precedence merging
├── cli_common.py         # Shared CLI helpers / path + metadata utilities
├── cli_pipeline.py       # Full per-recording processing orchestration
├── cli_metrics.py        # Metrics-only stage (reads BIDS derivatives)
├── cli_summary.py        # Single-file summarize/inspect stage
├── cli_figures.py        # Figure-plotting stage
├── config.py             # YAML/JSON config loader + validation
├── schema.py             # Event-code decoding / token-map schema
├── bids.py               # BIDS discovery, derivatives, validation
├── inputs.py             # Recording discovery + legacy→BIDS conversion
├── io_brainvision.py     # BrainVision I/O helpers
├── behavior.py           # Behavioral CSV parsing
├── align.py              # EEG ↔ behavioral alignment logic
├── epoching.py           # Epoch creation utilities
├── artifacts.py          # Blink + voltage artifact detection
├── ica.py                # ICA fitting and application
├── ica_diagnostics.py    # Blink diagnostics and ICA recommendation logic
├── evoked.py             # Evoked and grand-average helpers
├── gpu.py                # Optional GPU (MNE CUDA / CuPy) backend
├── qc.py                 # QC summary writer
├── analysis_runner.py    # Flat-directory post-hoc metrics engine (DEPRECATED; see note below)
├── metrics/
│   ├── erp.py            # ERP windowed metrics
│   ├── erp_timeseries.py # ERP time-series metrics (Parquet)
│   ├── erp_windows.py    # Canonical ERP window definitions
│   ├── io.py             # Epochs loaders (.fif, .set)
│   ├── writers.py        # CSV/Parquet output writers
│   └── tfr.py            # Time–frequency metrics
├── viz/
│   └── paper_figures.py  # Paper-ready plots from metrics outputs
└── oneclick/
    └── backend.py        # Local HTTP backend for the Electron GUI prototype

scripts/                          # Thin wrappers / legacy utilities (not installed)
├── process_eeg_data.py           # Wrapper: --process_data
├── compute_eeg_metrics.py        # Wrapper: --get_metrics
├── plot_eeg_figures.py           # Wrapper: --plot_figures
├── export_eventcodes.py          # Legacy: export event codes
└── import_manual_rejection_sets.py  # Legacy: import EEGLAB manual rejections

run_analysis.py           # Compatibility shim for analysis_runner (eeg-run-analysis) — DEPRECATED
run_metrics.py            # Legacy metrics runner (kept for compatibility) — DEPRECATED
```

> **Deprecated: flat-directory metrics runners.** The `eeg-run-analysis`
> console script, the root `run_analysis.py` / `run_metrics.py` wrappers, and
> `python -m eeg_pipeline.analysis_runner` all emit a `DeprecationWarning` and
> will be removed in a future release. They read loose `*-epo.fif` files from a
> flat directory; prefer the config-driven, BIDS-based
> [`python -m eeg_pipeline.cli --get_metrics`](#metrics-only-reruns), which
> routes through the same `eeg_pipeline.metrics` engine (identical ERP/TFR
> numerics and a single TFR-baseline policy).

---

## Configuration-driven workflow

All pipeline behavior is controlled via a **single YAML (or JSON) config file**.
CLI flags are intentionally minimal.

Current contract:

- Default input is an existing **BIDS EEG dataset**.
- Legacy lab-layout input is still supported via `--legacy`.
- Optional legacy-to-BIDS conversion is available via `--convert_to_bids`.
- Output is written as a **BIDS derivatives dataset** under `derivatives/eeg-pipeline/`.

### Example `config.yaml`

```yaml
input:
  mode: bids

paths:
  bids_root: /data/bids_eeg
  derivatives_root: /data/bids_eeg/derivatives
  sourcedata_root: /data/bids_eeg/sourcedata
  raw_dir: null
  subject_csv_dir: null

bids:
  tasks: [oddball]

channels:
  eog_chs: []
  blink_proxy_chs: [Fp1]

preprocess:
  montage: standard_1020
  reref: average     # average | none
  l_freq: 0.1
  h_freq: 30
  notch_hz: [60]

events:
  # Primary contract: read source *_events.tsv and sidecars.
  # Optional fallback only for legacy imports when an events.tsv is missing.
  csv_fallback_dir: null
  behavioral_keep_codes: [110, 111, 210, 211]
  standard_codes: [110, 210]
  deviant_codes: [111, 211]

conversion:
  enabled: false
  bids_output_root: null
  overwrite: true

epoching:
  tmin: -0.2
  tmax: 0.6
  baseline: [-0.2, 0.0]

ica:
  mode: auto
  auto_blink_rate_per_min: 15
  method: fastica
  n_components: 0.99
  save_ica: true

metrics:
  erp:
    enabled: true
    # Label for Deviant-Standard difference wave (optional)
    difference_label: DEV_MINUS_STD
    windows:
      - name: MMN_150_250
        tmin: 0.15
        tmax: 0.25
    timeseries: true

  tfr:
    enabled: true
    tmin: -0.2
    tmax: 0.6
    fmin: 3.0
    fmax: 8.0
    time_decim: 1
    # TF-domain baseline is intentionally not applied (ERPLAB-equivalent)
```

## Running the Pipeline

From the repository root (module invocation):

```bash
python -m eeg_pipeline.cli --config config.yaml --process_data --get_metrics
```

### OneClick Electron GUI prototype

This branch includes an early Electron + Python-backend GUI shell. It keeps the
existing CLI as the execution engine and adds a local backend for validation,
recording discovery, run launch, and log polling.

Install the Electron development dependency, then launch the app:

```bash
npm install
npm run oneclick
```

The backend can also be started directly for API testing:

```bash
python3 -m eeg_pipeline.oneclick.backend
```

The GUI reads the selected config file, validates it with the same config loader
used by the CLI, discovers BIDS/legacy recordings, and starts the existing
pipeline in a subprocess. For this prototype, the config file must be named
`config.yaml`, `config.yml`, or `config.json` and live at the repository root.

If you omit the stage flags, the default is `--process_data --get_metrics`.

Legacy layout input is opt-in:

```bash
python -m eeg_pipeline.cli \
  --config config.yaml \
  --legacy \
  --raw_dir /data/legacy_raw \
  --subject_csv_dir /data/legacy_behavior \
  --process_data --get_metrics
```

To convert a legacy dataset to BIDS without running the rest of the pipeline:

```bash
python -m eeg_pipeline.cli \
  --config config.yaml \
  --legacy \
  --raw_dir /data/legacy_raw \
  --subject_csv_dir /data/legacy_behavior \
  --convert_to_bids \
  --conversion_bids_root /data/legacy_bids
```

### ERP CORE preset (optional)

You can enable an [ERP CORE‑style preset](https://doi.org/10.1016/j.neuroimage.2020.117465) (Kappenman et al., 2021) via `--erp-core`. This applies the following defaults:

- `preprocess.reref = tp9_tp10`
- `preprocess.l_freq = 0.1`
- `preprocess.h_freq = 20.0`
- `artifacts.voltage.method = simple`
- `artifacts.voltage.auto_percentile = 97.5`
- `artifacts.blink.auto_percentile = 99.0`
- `ica = on`

Example:

```bash
python -m eeg_pipeline.cli \
  --config config.yaml \
  --erp-core \
  --process_data --get_metrics
```

CLI flags still override these defaults if explicitly provided.

Optional debugging / inspection of a single file:

```bash
python -m eeg_pipeline.cli \
  --config config.yaml \
  --summarize_one_file /data/bids_eeg/sub-01/eeg/sub-01_task-oddball_run-01_eeg.vhdr
```

EEGLAB example:

```bash
python -m eeg_pipeline.cli \
  --config config.yaml \
  --summarize_one_file /path/to/sub-1001_task-WordPR_eeg.set
```

### Opinionated wrappers (optional)

```bash
python scripts/process_eeg_data.py --config config.yaml
python scripts/compute_eeg_metrics.py --config config.yaml
python scripts/plot_eeg_figures.py --config config.yaml
```

## GPU Acceleration (Optional)

Enable GPU acceleration to speed up supported steps.

- Config (recommended): add `compute.use_gpu: true` and optionally `compute.gpu_device: 0`.
- CLI: pass `--use_gpu` and optionally `--gpu_device 0`.
- Behavior: the pipeline attempts to initialize MNE CUDA (if available) and uses CuPy for internal array operations (artifact rejection). If GPU libraries are missing, it falls back to CPU and prints a warning.

## Metrics-only reruns

The metrics stage now reads derivative epoch files from the BIDS derivatives tree:

```bash
python -m eeg_pipeline.cli \
  --config config.yaml \
  --get_metrics
```

## HPC / SLURM (parallel processing)

Subjects are processed independently, so the largest speedup comes from running
them concurrently rather than from more cores per subject. Each recording writes
self-contained per-subject derivatives, and a separate **gather** step rebuilds
the dataset-level tables and grand averages from those files.

```bash
ls -d /data/bids_eeg/sub-* | xargs -n1 basename > subjects.txt
./hpc/submit.sh config.yaml subjects.txt 10
```

That submits one array task per subject plus a dependent gather job. Equivalently,
by hand:

```bash
# 1. per subject (one array task each)
python -m eeg_pipeline.cli --config config.yaml \
  --process_data --get_metrics --subjects sub-01 --n_jobs 4

# 2. once all subjects finish, gather dataset-level outputs
python -m eeg_pipeline.cli --config config.yaml --aggregate_only
```

`--aggregate_only` reads only what is already on disk, never reprocesses, and
overwrites its outputs — so it is safe to re-run after a partial array. It reports
how many per-subject files it found; compare that against your expected subject
count to catch tasks that failed.

### Two axes of parallelism

| Axis | Mechanism | Speedup |
| --- | --- | --- |
| **Across participants** | SLURM array, one task per subject | Near-linear — bounded by concurrent slots |
| **Within subject** (many channels) | `--n_jobs` / `compute.n_jobs` | Sublinear — see below |

`--n_jobs` parallelizes the MNE operations that split across channels: filtering,
notch, and `compute_tfr`. It does **not** speed up `ICA.fit`, which takes no
`n_jobs` and is effectively serial for fastica/infomax/picard. When ICA dominates
per-subject time, `n_jobs` yields well under linear returns while array
parallelism stays near-linear — so on a fixed core budget, prefer **more array
tasks over more cores per task**.

The QC summary records per-stage wall clock (`t_preprocess_s`, `t_ica_s`,
`t_epoching_s`, `t_metrics_s`, `t_io_s`), so one real run tells you your actual
ICA-vs-TFR split and therefore whether raising `--n_jobs` is worth it.

### Two ways to lose the speedup

- **BLAS oversubscription.** NumPy/MNE spawn threaded BLAS by default. Many
  concurrent tasks each spawning one thread per core will thrash and can run
  *slower* than serial. Set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` /
  `OPENBLAS_NUM_THREADS` to `$SLURM_CPUS_PER_TASK` — the shipped templates do this.
- **Memory, not cores, is usually the binding constraint.** Each task holds its
  raw and epochs in RAM, so concurrency is typically capped by `--mem` before
  cores run out. Throttle with `%N` on `--array` if tasks are being OOM-killed.

GPU (`compute.use_gpu`) accelerates FFT-based filtering and resampling only —
MNE's CUDA support does not cover ICA or time-frequency decomposition, so a
`--gres=gpu:1` request buys comparatively little here.

## Visualization (paper_figures)

```bash
python -m eeg_pipeline.viz.paper_figures \
  --erp_parquet /data/bids_eeg/derivatives/eeg-pipeline/eeg/desc-erp_timeseries.parquet \
  --tfr_file /data/bids_eeg/derivatives/eeg-pipeline/eeg/desc-tfr_metrics.tsv \
  --out_dir /path/to/figures \
  --time_window 0.15 0.25 \
  --freq_band 3 8 \
  --diff_heatmap
```

## Outputs

```
derivatives/
└── eeg-pipeline/
    ├── dataset_description.json
    ├── eeg/
    │   ├── desc-summary_qc.tsv
    │   ├── desc-erp_metrics.tsv
    │   ├── desc-tfr_metrics.tsv
    │   ├── desc-erp_timeseries.parquet
    │   └── task-<task>_desc-grandaverage-<condition>_ave.fif
    └── sub-<id>/
        └── [ses-<id>/]eeg/
            ├── sub-<id>[_ses-<id>]_task-<task>[_run-<run>]_desc-preproc_eeg.fif
            ├── sub-<id>[_ses-<id>]_task-<task>[_run-<run>]_epo.fif
            ├── sub-<id>[_ses-<id>]_task-<task>[_run-<run>]_desc-aligned_events.tsv
            ├── sub-<id>[_ses-<id>]_task-<task>[_run-<run>]_desc-standard_ave.fif
            ├── sub-<id>[_ses-<id>]_task-<task>[_run-<run>]_desc-deviant_ave.fif
            ├── sub-<id>[_ses-<id>]_task-<task>[_run-<run>]_desc-erp_metrics.tsv
            ├── sub-<id>[_ses-<id>]_task-<task>[_run-<run>]_desc-tfr_metrics.tsv
            └── matching JSON sidecars for provenance
```

Each derivative file gets a JSON sidecar with filtering, rereferencing, epoch window, artifact rejection, ICA settings, and source-file references. The aligned event export is written in BIDS tabular form as `*_events.tsv` plus `*_events.json`.

## Validation

The test suite now includes a tiny BIDS fixture and validates both:

- the input dataset structure, and
- the produced derivatives tree

using the repo's BIDS-aware structural validator helpers in `eeg_pipeline.bids`.

## Design philosophy
- Explicit over implicit: no hidden heuristics
- Fail loudly, skip safely: broken subjects don’t crash batch runs
- MNE-native: outputs are standard FIF objects
- MATLAB-aware: folder structure and logic map cleanly to common EEGLAB workflows


⸻

## Requirements

**Core** (required):
- Python ≥ 3.10
- mne
- numpy
- scipy
- pandas
- pyarrow (Parquet)
- pyyaml (for YAML configs)

**Optional extras** (install only if needed):
- `viz` — matplotlib, seaborn (paper figures / `--plot_figures`)
- `gpu` — a CUDA-compatible CuPy build
- `ica` — python-picard (for `ica.method: picard`)

Install:
```bash
pip install ".[viz]"        # core + visualization (typical)
pip install ".[dev,viz]"    # add the test/lint toolchain
# or, for a plain convenience install:
pip install -r requirements.txt
```

The authoritative dependency contract (with version bounds) lives in
`pyproject.toml`.

## Acknowledgments

Portions of this pipeline were developed with the assistance of **ChatGPT (GPT-5.2, OpenAI)**, which was used as an interactive programming and design aid.  
All scientific decisions, methodological choices, and final code were reviewed and implemented by the author.

## Attribution/Citation

If you use this pipeline in data processing, please consider citing the package:

`Berry, G. M. (2026). eeg-pipeline (v2.1.0). Zenodo. https://doi.org/10.5281/zenodo.18704504`
