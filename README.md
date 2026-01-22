# eeg-pipeline

A modular, config-driven EEG preprocessing and analysis pipeline built on **MNE-Python**.  
Originally developed for MMN paradigms, the pipeline is now **task-agnostic** and supports
ERP and time–frequency analyses across arbitrary experimental designs.

This project is designed for **research-grade EEG workflows** with an emphasis on:

- Reproducibility
- Auditability (QC summaries, ICA diagnostics)
- MATLAB → MNE conceptual continuity
- Scalable batch processing

---

## Key Features

### Core preprocessing
- BrainVision (.vhdr / .vmrk) input
- Standard montages (e.g., `standard_1020`)
- Average rereferencing
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
- Optional channel-level ERP time series extraction (planned / in progress)

### Time–frequency analysis (optional)
- Evoked TFR computation (multitaper or Morlet)
- Configurable frequency ranges, baselines, and time windows
- Fully compatible with MNE `AverageTFR` objects

### Quality control
- Per-subject QC rows written to `qc_summary.csv`
- Includes:
  - Event counts
  - Epoch rejection rates
  - Blink metrics
  - ICA decisions and exclusions
- Designed to support downstream statistical screening

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
│   ├── erp.py            # ERP metrics (windowed + time series)
│   └── tfr.py            # Time–frequency metrics
└── qc.py                 # QC summary writer
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
    windows:
      - name: MMN_150_250
        tmin: 0.15
        tmax: 0.25
    timeseries: false

  tfr:
    enabled: false
```

## Running the Pipeline

From the repository root:

```bash
python run_eeg_pipeline.py --config config.yaml
```

Optional debugging / inspection of a single file:

```bash
python run_eeg_pipeline.py \
  --config config.yaml \
  --summarize_one_file /path/to/S203.vhdr
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
└── qc_summary.csv          # Per-subject QC table
```

## Design philosophy
- Explicit over implicit: no hidden heuristics
- Fail loudly, skip safely: broken subjects don’t crash batch runs
- MNE-native: outputs are standard FIF objects
- MATLAB-aware: folder structure and logic map cleanly to common EEGLAB workflows


## Roadmap
- ERP peak latency and amplitude metrics
- Channel clusters and ROI definitions
- Trial-level TFR metrics
- BIDS-compatible export
- Automated report generation

⸻

## Requirements
- Python ≥ 3.10
- mne
- numpy
- pandas
- pyyaml (for YAML configs)
