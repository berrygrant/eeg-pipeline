#!/bin/bash
# Submit the per-subject array plus its dependent gather job.
#
#   ./hpc/submit.sh config.yaml subjects.txt [MAX_CONCURRENT]
#
# Builds subjects.txt from the BIDS root first if it does not exist. The gather
# job is chained with afterok, so it runs only if every array task succeeded --
# rerun it manually with `sbatch hpc/slurm_gather.sbatch` if you would rather
# aggregate whatever finished.
set -euo pipefail

CONFIG="${1:-config.yaml}"
SUBJECT_LIST="${2:-subjects.txt}"
MAX_CONCURRENT="${3:-10}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi
if [[ ! -f "$SUBJECT_LIST" ]]; then
    echo "Subject list not found: $SUBJECT_LIST" >&2
    echo "Create one with the subject labels to process, one per line, e.g.:" >&2
    echo "  ls -d /path/to/bids/sub-* | xargs -n1 basename > $SUBJECT_LIST" >&2
    exit 1
fi

# Count non-blank lines; slurm_array.sbatch indexes non-blank lines too, so the
# two stay in step even if the list has blank or trailing lines.
N_SUBJECTS="$(grep -c '[^[:space:]]' "$SUBJECT_LIST")"
if [[ "$N_SUBJECTS" -eq 0 ]]; then
    echo "Subject list is empty: $SUBJECT_LIST" >&2
    exit 1
fi

mkdir -p logs

echo "Submitting ${N_SUBJECTS} subject task(s), up to ${MAX_CONCURRENT} at a time."
ARRAY_JOBID="$(
    CONFIG="$CONFIG" SUBJECT_LIST="$SUBJECT_LIST" sbatch \
        --parsable \
        --array="0-$((N_SUBJECTS - 1))%${MAX_CONCURRENT}" \
        --export=ALL,CONFIG="$CONFIG",SUBJECT_LIST="$SUBJECT_LIST" \
        hpc/slurm_array.sbatch
)"
echo "  array job: ${ARRAY_JOBID}"

GATHER_JOBID="$(
    sbatch --parsable \
        --dependency="afterok:${ARRAY_JOBID}" \
        --export=ALL,CONFIG="$CONFIG" \
        hpc/slurm_gather.sbatch
)"
echo "  gather job: ${GATHER_JOBID} (runs after the array succeeds)"
