# eeg_pipeline/metrics/__init__.py
"""
Postprocessing metrics for ERP and time-frequency analyses.

Designed to run on:
- MNE FIF epochs: *-epo.fif (recommended; produced by pipeline)
- EEGLAB epochs: *.set
"""

from .io import load_epochs
from .erp import compute_erp_metrics
from .tfr import compute_tfr_metrics

__all__ = ["load_epochs", "compute_erp_metrics", "compute_tfr_metrics"]