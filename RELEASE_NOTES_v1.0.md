# v1.0

## Summary
v1.0 is a major release that adds GPU acceleration, EEGLAB support, expanded configuration/epoching/artifact controls, and new metrics/visualization tooling. It also aligns ERP/TFR behavior with ERPLAB conventions and improves metadata handling.

## Highlights
- GPU acceleration support (CuPy/MNE CUDA).
- EEGLAB `.set` ingestion alongside BrainVision `.vhdr`.
- Configurable condition maps and windowed artifact rejection.
- New analysis and visualization tooling (metrics + paper figures).

## Changes
- GPU support added, including runtime configuration and capability reporting.
- EEGLAB `.set` import support and hardened raw ingestion.
- ERP behavior updates: difference‑wave labeling clarified and ERP/TFR alignment with ERPLAB.
- Condition‑based filtering helpers and metadata derivation from condition maps.
- Windowed voltage artifact rejection parameters (method, threshold, window/step).
- Optional `max_reject_rate` safeguard for artifact rejection.
- Channel normalization for Fp1/Fp2 to match montage expectations.
- Added ERP CORE preset mode.
- New `run_analysis.py` for post‑hoc metrics.
- New `viz/paper_figures.py` and related figure outputs.
- CLI entrypoint fixes and robustness improvements.
- Updated requirements (including CuPy).

## Validation
- GPU validation completed on Thu Feb 19 11:31:52 2026.
- Pipeline run with `--use_gpu` completed successfully on RTX 3090.

## Environment
- Python: 3.10.19  
- MNE: 1.11.0  
- NumPy: 2.2.6  
- SciPy: 1.15.2  
- CuPy: 14.0.0  
- GPU driver: 570.211.01  
- CUDA: 12.8  
- GPU: NVIDIA GeForce RTX 3090  

## Known Issues
- None reported at release time.

