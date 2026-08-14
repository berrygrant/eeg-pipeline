#!/bin/bash
# Per-subject processing for a validation run: one array task per subject.
#
# Submitted by hpc/submit_validation.sh, which runs the precheck first (freezing
# provenance and writing SUBJECT_LIST) and chains the gather after this.
#
# Sized by NON-BLANK line count, matching how this script resolves each task:
#   --array=0-$(($(grep -c '[^[:space:]]' subjects.txt) - 1))%10
# Sizing: one task covers ONE SUBJECT, which in a multi-task dataset means every
# recording that subject has -- ds003620 has six (lab/oval/campus x count/ignore),
# so one task is six full preprocess+ICA passes, not one. Nothing here resamples
# either (no resample option exists), so ICA runs at the native rate. Budget for
# the whole subject, not for one recording.
#
# Override per dataset without editing this file:
#   VALIDATE_TIME=24:00:00 VALIDATE_MEM=64G ./hpc/submit_validation.sh ...
#SBATCH --job-name=eeg-validate
#SBATCH --output=logs/validate-%A_%a.out
#SBATCH --error=logs/validate-%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

CONFIG="${CONFIG:?CONFIG not set}"
BIDS_ROOT="${BIDS_ROOT:?BIDS_ROOT not set}"
OUT_DIR="${OUT_DIR:?OUT_DIR not set}"
SUBJECT_LIST="${SUBJECT_LIST:?SUBJECT_LIST not set}"

# Threaded BLAS spawns one thread per core inside EVERY task. With many
# concurrent tasks that oversubscribes the node and can run slower than serial.
# Pin BLAS to the cores SLURM actually granted this task.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Matplotlib must not try to reach a display on a compute node.
export MPLBACKEND=Agg

# Index NON-BLANK lines so a blank or trailing line cannot desynchronize the
# task id from the subject, which would land a task on an empty string and
# silently process nobody while reporting a task for that index.
SUBJECT="$(grep -v '^[[:space:]]*$' "$SUBJECT_LIST" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")"
if [[ -z "$SUBJECT" ]]; then
    echo "No subject at index ${SLURM_ARRAY_TASK_ID} of $SUBJECT_LIST" >&2
    echo "(the array size must match the number of non-blank lines)" >&2
    exit 1
fi

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${SUBJECT} on $(hostname) ==="
echo "started: $(date -Is)"
# Record the limits in the log. When a task is killed the log is the only place
# that says what it was killed against, and "OUT_OF_MEMORY at 32G" is a different
# problem from "OUT_OF_MEMORY at 4G".
echo "limits:  cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_CPU:-?}} time=$(squeue -h -j "${SLURM_JOB_ID}" -o %l 2>/dev/null || echo '?')"

# --skip_aggregate is REQUIRED. Dataset-level tables and grand averages are
# shared paths; without it every task would rebuild them concurrently, racing one
# another and reading files other tasks are still writing. The gather builds them
# once, after all tasks finish.
python -m eeg_pipeline.cli \
    --config "$CONFIG" \
    --bids_root "$BIDS_ROOT" \
    --derivatives_root "${OUT_DIR}/derivatives" \
    --process_data \
    --get_metrics \
    --subjects "$SUBJECT" \
    --skip_aggregate \
    --n_jobs "${SLURM_CPUS_PER_TASK:-1}"

echo "finished: $(date -Is)"
