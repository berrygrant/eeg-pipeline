#!/bin/bash
# Submit a full validation run: precheck -> per-subject array -> gather/package.
#
#   ./hpc/submit_validation.sh <config> <bids-root> <out-dir> <dataset-id> <dataset-version> [MAX_CONCURRENT]
#
# Example:
#   ./hpc/submit_validation.sh config.yaml \
#       /scratch/$USER/ds003620 \
#       /scratch/$USER/runs/ds003620_$(date +%F) \
#       ds003620 1.1.1 20
#
# The precheck runs HERE on the login node (it is seconds of filesystem
# checking, no compute) so that a bad path or a failed integrity gate is caught
# before anything is queued. It also freezes provenance and writes the subject
# list the array indexes.
set -euo pipefail

CONFIG="${1:?usage: submit_validation.sh <config> <bids-root> <out-dir> <dataset-id> <dataset-version> [MAX_CONCURRENT]}"
BIDS_ROOT="${2:?bids-root required}"
OUT_DIR="${3:?out-dir required}"
DATASET_ID="${4:-unknown}"
DATASET_VERSION="${5:-unknown}"
MAX_CONCURRENT="${6:-10}"

# ---------------------------------------------------------------------------
# Site settings. These differ per cluster -- set them here or export them first.
# SLURM_ACCOUNT/SLURM_PARTITION are left empty by default because a wrong value
# fails at submit time with a clearer message than a guessed one would.
# ---------------------------------------------------------------------------
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_MAIL_USER="${SLURM_MAIL_USER:-}"

EXTRA_SBATCH=()
[[ -n "$SLURM_ACCOUNT"   ]] && EXTRA_SBATCH+=(--account="$SLURM_ACCOUNT")
[[ -n "$SLURM_PARTITION" ]] && EXTRA_SBATCH+=(--partition="$SLURM_PARTITION")
[[ -n "$SLURM_MAIL_USER" ]] && EXTRA_SBATCH+=(--mail-user="$SLURM_MAIL_USER" --mail-type=END,FAIL)

command -v sbatch >/dev/null || { echo "sbatch not found -- run this on the cluster, not your laptop." >&2; exit 1; }

CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
BIDS_ROOT="$(cd "$BIDS_ROOT" && pwd)"
mkdir -p "$OUT_DIR" logs
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

SUBJECT_LIST="${OUT_DIR}/subjects.txt"

echo "== precheck (gates 0-1) =="
python scripts/validate_dataset.py \
    --stage precheck \
    --config "$CONFIG" \
    --bids-root "$BIDS_ROOT" \
    --out-dir "$OUT_DIR" \
    --dataset-id "$DATASET_ID" \
    --dataset-version "$DATASET_VERSION" \
    --retrieved "$(date +%F)" \
    --subject-list "$SUBJECT_LIST" \
    ${ALLOW_INTEGRITY_WARNINGS:+--allow-integrity-warnings}

# Non-blank count, matching how validate_array.sbatch indexes the list.
# `|| true` because grep -c exits 1 on a zero count, which under `set -e` would
# abort before the explicit message below.
N_SUBJECTS="$(grep -c '[^[:space:]]' "$SUBJECT_LIST" || true)"
if [[ "$N_SUBJECTS" -eq 0 ]]; then
    echo "No subjects found under $BIDS_ROOT" >&2
    exit 1
fi

echo
echo "== submitting =="
echo "  subjects:   ${N_SUBJECTS} (up to ${MAX_CONCURRENT} concurrent)"
echo "  out-dir:    ${OUT_DIR}"

ARRAY_JOBID="$(
    sbatch --parsable \
        "${EXTRA_SBATCH[@]}" \
        --array="0-$((N_SUBJECTS - 1))%${MAX_CONCURRENT}" \
        --export=ALL,CONFIG="$CONFIG",BIDS_ROOT="$BIDS_ROOT",OUT_DIR="$OUT_DIR",SUBJECT_LIST="$SUBJECT_LIST" \
        hpc/validate_array.sbatch
)"
echo "  array job:  ${ARRAY_JOBID}"

GATHER_JOBID="$(
    sbatch --parsable \
        "${EXTRA_SBATCH[@]}" \
        --dependency="afterok:${ARRAY_JOBID}" \
        --export=ALL,CONFIG="$CONFIG",BIDS_ROOT="$BIDS_ROOT",OUT_DIR="$OUT_DIR",DATASET_ID="$DATASET_ID",DATASET_VERSION="$DATASET_VERSION" \
        hpc/validate_gather.sbatch
)"
echo "  gather job: ${GATHER_JOBID} (runs only if every array task succeeds)"
echo
echo "Watch:    squeue -u \$USER"
echo "Package:  ${OUT_DIR}"
echo
echo "If some array tasks fail, the gather will not run. Aggregate what landed with:"
echo "  CONFIG='$CONFIG' BIDS_ROOT='$BIDS_ROOT' OUT_DIR='$OUT_DIR' \\"
echo "    DATASET_ID='$DATASET_ID' DATASET_VERSION='$DATASET_VERSION' \\"
echo "    sbatch ${EXTRA_SBATCH[*]} hpc/validate_gather.sbatch"
echo "participant_flow.csv will then name exactly who is missing."
