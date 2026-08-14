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

# Per-task resources. The #SBATCH lines in validate_array.sbatch are the
# defaults; these override them on the command line, which SLURM honours over
# the in-file directives. Raise them when tasks come back OUT_OF_MEMORY or
# TIMEOUT rather than editing the template.
VALIDATE_TIME="${VALIDATE_TIME:-}"
VALIDATE_MEM="${VALIDATE_MEM:-}"
VALIDATE_CPUS="${VALIDATE_CPUS:-}"

EXTRA_SBATCH=()
[[ -n "$SLURM_ACCOUNT"   ]] && EXTRA_SBATCH+=(--account="$SLURM_ACCOUNT")
[[ -n "$SLURM_PARTITION" ]] && EXTRA_SBATCH+=(--partition="$SLURM_PARTITION")
[[ -n "$SLURM_MAIL_USER" ]] && EXTRA_SBATCH+=(--mail-user="$SLURM_MAIL_USER" --mail-type=END,FAIL)

ARRAY_SBATCH=()
[[ -n "$VALIDATE_TIME" ]] && ARRAY_SBATCH+=(--time="$VALIDATE_TIME")
[[ -n "$VALIDATE_MEM"  ]] && ARRAY_SBATCH+=(--mem="$VALIDATE_MEM")
[[ -n "$VALIDATE_CPUS" ]] && ARRAY_SBATCH+=(--cpus-per-task="$VALIDATE_CPUS")

command -v sbatch >/dev/null || { echo "sbatch not found -- run this on the cluster, not your laptop." >&2; exit 1; }

# Expand a leading ~ ourselves. Bash expands it on the command line only when
# unquoted, so a quoted "~/scratch/ds003620" -- the natural way to write a path
# that might contain spaces -- would otherwise arrive here as a literal tilde and
# fail as a missing directory.
CONFIG="${CONFIG/#\~/$HOME}"
BIDS_ROOT="${BIDS_ROOT/#\~/$HOME}"
OUT_DIR="${OUT_DIR/#\~/$HOME}"

# Check the inputs before resolving them. Bare `cd` under `set -e` aborts with a
# terse "cd: ...: No such file or directory" that names the line number rather
# than the problem, which is a poor first thing to hit on a cluster.
if [[ ! -f "$CONFIG" ]]; then
    echo "Config file not found: $CONFIG" >&2
    echo "  (paths are resolved from $(pwd))" >&2
    exit 1
fi
if [[ ! -d "$BIDS_ROOT" ]]; then
    echo "BIDS root not found: $BIDS_ROOT" >&2
    echo "  The dataset has to exist on the cluster before submitting. If you have" >&2
    echo "  not staged it yet, copy it up or fetch it directly onto \$SCRATCH." >&2
    exit 1
fi
if ! compgen -G "${BIDS_ROOT}/sub-*" >/dev/null; then
    echo "No sub-* directories under: $BIDS_ROOT" >&2
    echo "  --bids-root must be the directory CONTAINING sub-01/, sub-02/, ..." >&2
    echo "  Found instead:" >&2
    ls -1 "$BIDS_ROOT" 2>/dev/null | head -5 | sed 's/^/    /' >&2
    echo "  If the dataset is nested, point at the inner directory." >&2
    exit 1
fi
if ! mkdir -p "$OUT_DIR" logs 2>/dev/null; then
    echo "Cannot create output directory: $OUT_DIR" >&2
    echo "  Check the path exists up to its parent and that you can write there." >&2
    exit 1
fi

CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
BIDS_ROOT="$(cd "$BIDS_ROOT" && pwd)"
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
        "${EXTRA_SBATCH[@]}" "${ARRAY_SBATCH[@]}" \
        --array="0-$((N_SUBJECTS - 1))%${MAX_CONCURRENT}" \
        --export=ALL,CONFIG="$CONFIG",BIDS_ROOT="$BIDS_ROOT",OUT_DIR="$OUT_DIR",SUBJECT_LIST="$SUBJECT_LIST" \
        hpc/validate_array.sbatch
)"
echo "  array job:  ${ARRAY_JOBID}"

# afterany, NOT afterok. Over real archival data some subjects are expected to
# fail -- a broken BrainVision link, events with no usable codes -- and naming
# them is the deliverable, not an accident. Under afterok a single such subject
# leaves the gather permanently PENDING as DependencyNeverSatisfied and produces
# nothing at all, which is the least informative possible outcome. The gather
# tolerates partial input by design and exits non-zero if NOTHING landed, so a
# genuinely systemic failure still reads as a failure.
GATHER_JOBID="$(
    sbatch --parsable \
        "${EXTRA_SBATCH[@]}" \
        --dependency="afterany:${ARRAY_JOBID}" \
        --export=ALL,CONFIG="$CONFIG",BIDS_ROOT="$BIDS_ROOT",OUT_DIR="$OUT_DIR",DATASET_ID="$DATASET_ID",DATASET_VERSION="$DATASET_VERSION" \
        hpc/validate_gather.sbatch
)"
echo "  gather job: ${GATHER_JOBID} (runs once the array finishes, pass or fail)"
echo
echo "Watch:    squeue -u \$USER"
echo "Package:  ${OUT_DIR}"
echo
echo "participant_flow.csv will name every subject that did not make it, and why."
