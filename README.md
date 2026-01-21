# eeg-pipeline

A modular, reproducible EEG preprocessing and analysis pipeline built in Python using MNE. The pipeline is designed to mirror and extend common EEGLAB/ERPLAB workflows while remaining fully scriptable, auditable, and extensible across ERP and time–frequency paradigms.

## Motivation

EEGLAB and ERPLAB provide powerful GUI-based EEG analysis tools, but large-scale, multi-subject studies benefit from automated, version-controlled pipelines. `eeg-pipeline` provides a Python-native alternative that preserves the conceptual structure of EEGLAB/ERPLAB preprocessing (re-referencing, filtering, event lists, binning, epoching, artifact rejection, ICA) while enabling reproducibility, diagnostics, and flexible experimental designs.

## Key Features

- BrainVision (.vhdr/.vmrk) import via MNE
- Flexible behavioral event alignment from external CSV files
- Event-code schema parsing with extensible metadata derivation
- ERP-focused preprocessing (filtering, re-referencing, baseline correction)
- Artifact detection (blink and muscle) with configurable thresholds
- Proxy blink diagnostics for datasets without EOG channels
- ICA diagnostics and recommendation logic (optional, non-destructive)
- Automated QC summaries across subjects
- Modular architecture supporting non-MMN paradigms

## Pipeline Overview

The default pipeline mirrors a typical ERPLAB workflow:

1. Raw EEG import (BrainVision)
2. Channel typing, montage assignment, re-referencing
3. Notch and band-pass filtering
4. Event extraction and behavioral code alignment
5. Epoching and baseline correction
6. Artifact detection and rejection
7. Evoked response computation
8. Grand averaging across subjects
9. QC summary generation

All steps are configurable via command-line arguments.

## Repository Structure

```bash
eeg-pipeline/
├── run_mmn_pipeline.py
├── mmn_pipeline/
│   ├── cli.py
│   ├── io_brainvision.py
│   ├── behavior.py
│   ├── align.py
│   ├── epoching.py
│   ├── artifacts.py
│   ├── ica_diagnostics.py
│   ├── evoked.py
│   ├── schema.py
│   └── qc.py
└── README.md
```

The `mmn_pipeline` module contains reusable components; `run_mmn_pipeline.py` is a thin CLI entry point.

## Example Usage

Summarize a single file (sanity checks only):

```bash
python run_mmn_pipeline.py \
  --raw_dir path/to/raw_data \
  --subject_csv_dir path/to/behavioral_data \
  --out_dir /tmp/mmn_test \
  --summarize_one_file path/to/s203.vhdr
```
Run the full pipeline for a subject:

```bash
python run_mmn_pipeline.py \
  --raw_dir path/to/raw_data \
  --subject_csv_dir path/to/behavioral_data \
  --out_dir /tmp/mmn_results \
  --subjects s203 \
  --behavioral_keep_codes 110 111 210 211 \
  --token_map EH IH
```

## Event Code Schemas

The pipeline supports structured, multi-digit event codes (e.g., ABC), where each digit encodes experimental factors such as block type, buffer status, and condition. Metadata derived from event codes is attached to epochs and can be extended without modifying core pipeline logic.

## ICA Diagnostics

The pipeline includes optional ICA diagnostics that estimate blink and ocular artifact prevalence using either EOG channels or frontal EEG proxies. The diagnostics provide quantitative recommendations for whether ICA is warranted, without automatically modifying the data.

### Outputs

- Cleaned raw data (01_clean_raw/*.fif)
- Epoched data (02_epochs/*.fif)
- Subject-level evoked responses (03_evokeds/*.fif)
- Grand-average evoked responses (04_grand_averages/*.fif)
- QC summary CSV with alignment, artifact, and ICA metrics

# Status

This package is in active development. APIs may change substantially prior to a 1.0 release.