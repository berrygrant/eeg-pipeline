#!/bin/bash
# Gather + package a validation run, after every array task has finished.
#
# Runs --aggregate_only over whatever the array produced, then reconciles the
# participant flow and writes the evidence package (gates 3 and 5).
#
# Chained by hpc/submit_validation.sh with afterany, so it runs once the array
# finishes whether or not every task succeeded. That is deliberate: subjects that
# cannot be processed are a finding to report, not a reason to withhold the
# report. Aggregation overwrites its outputs, so re-running is safe, and it exits
# non-zero if nothing at all landed.
#
# To re-package an existing run by hand:
#
#   CONFIG=... BIDS_ROOT=... OUT_DIR=... sbatch hpc/validate_gather.sbatch
#SBATCH --job-name=eeg-validate-gather
#SBATCH --output=logs/validate-gather-%j.out
#SBATCH --error=logs/validate-gather-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

CONFIG="${CONFIG:?CONFIG not set}"
BIDS_ROOT="${BIDS_ROOT:?BIDS_ROOT not set}"
OUT_DIR="${OUT_DIR:?OUT_DIR not set}"
DATASET_ID="${DATASET_ID:-unknown}"
DATASET_VERSION="${DATASET_VERSION:-unknown}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLBACKEND=Agg

source "${SLURM_SUBMIT_DIR:-.}/hpc/env_setup.sh"

# Memory is the binding constraint here, not CPU: the gather concatenates every
# subject's tables and reads every evoked file to build the grand averages.
echo "=== gather on $(hostname) ==="
echo "started: $(date -Is)"

python scripts/validate_dataset.py \
    --stage package \
    --config "$CONFIG" \
    --bids-root "$BIDS_ROOT" \
    --out-dir "$OUT_DIR" \
    --dataset-id "$DATASET_ID" \
    --dataset-version "$DATASET_VERSION"

echo "finished: $(date -Is)"
