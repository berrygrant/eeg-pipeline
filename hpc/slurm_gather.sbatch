#!/bin/bash
# Gather step: rebuild dataset-level outputs from per-subject derivatives.
#
# Run after the per-subject array completes:
#
#   sbatch --dependency=afterok:<ARRAY_JOBID> hpc/slurm_gather.sbatch
#
# Reads the per-subject QC rows, metrics tables, and evoked files that each array
# task wrote, and produces the combined QC/metrics tables and grand averages.
# Single-threaded and I/O bound -- it concatenates tables and averages evokeds,
# so it needs one core and enough memory to hold the combined tables.
#
# Safe to re-run: every output is overwritten from whatever is on disk, so a
# retry after a partial array will not duplicate rows. If some tasks failed,
# check the reported file counts against your expected subject count -- the
# gather step aggregates whatever finished rather than failing.
#SBATCH --job-name=eeg-gather
#SBATCH --output=logs/eeg-gather-%j.out
#SBATCH --error=logs/eeg-gather-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

CONFIG="${CONFIG:-config.yaml}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Activate your environment here (edit to match your cluster), e.g.:
#   module load miniconda && conda activate eeg-pipeline
#   source /path/to/venv/bin/activate

echo "[$(date -Is)] gathering dataset-level outputs"

python -m eeg_pipeline.cli --config "$CONFIG" --aggregate_only
