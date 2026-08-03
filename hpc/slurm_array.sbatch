#!/bin/bash
# Per-subject processing as a SLURM array: one task per subject.
#
#   sbatch --array=0-$(($(wc -l < subjects.txt) - 1))%10 hpc/slurm_array.sbatch
#
# Then gather the dataset-level outputs once the array finishes:
#
#   sbatch --dependency=afterok:<ARRAY_JOBID> hpc/slurm_gather.sbatch
#
# Subjects are independent, so this is the main speedup: wall clock falls to
# roughly one subject plus the gather, bounded by how many tasks run at once.
#SBATCH --job-name=eeg-array
#SBATCH --output=logs/eeg-%A_%a.out
#SBATCH --error=logs/eeg-%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

CONFIG="${CONFIG:-config.yaml}"
SUBJECT_LIST="${SUBJECT_LIST:-subjects.txt}"

# Threaded BLAS will happily spawn one thread per core *inside every task*. With
# many concurrent tasks that oversubscribes the node and can run slower than
# serial. Pin BLAS to the cores SLURM actually gave this task.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Activate your environment here (edit to match your cluster), e.g.:
#   module load miniconda && conda activate eeg-pipeline
#   source /path/to/venv/bin/activate

SUBJECT="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBJECT_LIST")"
if [[ -z "$SUBJECT" ]]; then
    echo "No subject at line $((SLURM_ARRAY_TASK_ID + 1)) of $SUBJECT_LIST" >&2
    exit 1
fi

echo "[$(date -Is)] task ${SLURM_ARRAY_TASK_ID}: ${SUBJECT} (cpus=${SLURM_CPUS_PER_TASK:-1})"

# --n_jobs matches --cpus-per-task so MNE's channel-parallel operations use the
# allocation without exceeding it.
#
# --skip_aggregate is REQUIRED here. Dataset-level tables and grand averages are
# shared paths, so if every task aggregated at the end they would race each other
# and read files other tasks are still writing. The dependent gather job builds
# them once, after all tasks finish.
python -m eeg_pipeline.cli \
    --config "$CONFIG" \
    --process_data \
    --get_metrics \
    --subjects "$SUBJECT" \
    --skip_aggregate \
    --n_jobs "${SLURM_CPUS_PER_TASK:-1}"
