# eeg-pipeline

A modular, config-driven EEG preprocessing and analysis pipeline built on **MNE-Python**.  
Originally developed for MMN paradigms, the pipeline is now **task-agnostic** and supports
ERP and time–frequency analyses across arbitrary experimental designs.

This project is designed for **research-grade EEG workflows** with an emphasis on:

- Reproducibility
- Auditability (QC summaries, ICA diagnostics)
- MATLAB → MNE conceptual continuity
- Scalable batch processing

Current Release: [![DOI](https://zenodo.org/badge/1139314445.svg)](https://doi.org/10.5281/zenodo.18704504)
CodeCov:
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
├── cli.py                # Main pipeline entry point
├── config.py             # YAML/JSON config loader + validation
├── io_brainvision.py     # BrainVision I/O helpers
├── behavior.py           # Behavioral CSV parsing
├── align.py              # EEG ↔ behavioral alignment logic
├── epoching.py           # Epoch creation utilities
├── artifacts.py          # Blink + voltage artifact detection
├── ica.py                # ICA fitting and application
├── ica_diagnostics.py    # Blink diagnostics and ICA recommendation logic
├── evoked.py             # Evoked and grand-average helpers
├── metrics/
│   ├── erp.py            # ERP windowed metrics
│   ├── erp_timeseries.py # ERP time-series metrics (Parquet)
│   ├── erp_windows.py    # Canonical ERP window definitions
│   ├── io.py             # Epochs loaders (.fif, .set)
│   └── tfr.py            # Time–frequency metrics
├── viz/
│   └── paper_figures.py  # Paper-ready plots from metrics outputs
└── qc.py                 # QC summary writer

scripts/
├── process_eeg_data.py    # Process raw data → epochs/evokeds/QC
├── compute_eeg_metrics.py # Metrics from existing epochs
└── plot_eeg_figures.py    # Paper-ready figures from metrics outputs

run_analysis.py           # Post-hoc ERP/TFR metrics on epochs
run_metrics.py            # Legacy metrics runner (kept for compatibility)
```

---

## Configuration-driven workflow

All pipeline behavior is controlled via a **single YAML (or JSON) config file**.
CLI flags are intentionally minimal.

### Example `config.yaml`

```yaml
paths:
  raw_dir: /data/EEG/raw
  subject_csv_dir: /data/EEG/behavior
  out_dir: /data/EEG/derivatives

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
  behavioral_keep_codes: [110, 111, 210, 211]
  standard_codes: [110, 210]
  deviant_codes: [111, 211]

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

If you omit the stage flags, the default is `--process_data --get_metrics`.

### ERP CORE preset (optional)

You can enable an [ERP CORE‑style preset](https://doi.org/10.1016/j.neuroimage.2020.117465) (Kappeman et al., 2021) via `--erp-core`. This applies the following defaults:

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
  --summarize_one_file /path/to/S203.vhdr
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

## Post-hoc metrics (on existing epochs)

```bash
python run_analysis.py \
  --epochs_dir /path/to/02_epochs \
  --out_dir /path/to/05_metrics \
  --do_erp --do_tfr \
  --channels Fz Cz \
  --conditions Standard Deviant \
  --erp_window MMN 0.15 0.25 \
  --tfr_tmin -0.2 --tfr_tmax 0.6 \
  --tfr_fmin 3 --tfr_fmax 8 --tfr_fstep 1
```

## Visualization (paper_figures)

```bash
python -m eeg_pipeline.viz.paper_figures \
  --erp_parquet /path/to/05_metrics/erp_timeseries_all.parquet \
  --tfr_file /path/to/05_metrics/tfr_metrics_all.csv \
  --out_dir /path/to/figures \
  --time_window 0.15 0.25 \
  --freq_band 3 8 \
  --diff_heatmap
```

## Outputs

```
out_dir/
├── 00_ica/                 # Saved ICA objects (optional)
├── 01_clean_raw/           # Preprocessed raw FIF files
├── 02_epochs/              # Epoched data
├── 03_evokeds/             # Subject-level evoked responses
├── 04_grand_averages/      # Grand-average evokeds
├── 05_metrics/             # ERP / TFR metrics (if enabled)
│   ├── erp_metrics_all.csv
│   ├── tfr_metrics_all.csv
│   ├── erp_timeseries_all.parquet
│   └── erp_timeseries/      # per-subject Parquet time-series
└── qc_summary.csv          # Per-subject QC table
```

## Design philosophy
- Explicit over implicit: no hidden heuristics
- Fail loudly, skip safely: broken subjects don’t crash batch runs
- MNE-native: outputs are standard FIF objects
- MATLAB-aware: folder structure and logic map cleanly to common EEGLAB workflows


⸻

## Requirements
- Python ≥ 3.10
- mne
- numpy
- pandas
- pyarrow (Parquet)
- matplotlib
- seaborn
- pyyaml (for YAML configs)

Install:
```bash
pip install -r requirements.txt
```

## Acknowledgments

Portions of this pipeline were developed with the assistance of **ChatGPT (GPT-5.2, OpenAI)**, which was used as an interactive programming and design aid.  
All scientific decisions, methodological choices, and final code were reviewed and implemented by the author.

## Attribution/Citation

If you use this pipeline in data processing, please consider citing the package:

`Berry, G. M. (2026). eeg-pipeline (v1.0). Zenodo. https://doi.org/10.5281/zenodo.18704505`
