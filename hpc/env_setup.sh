# Shared environment setup for the validation sbatch templates. Sourced, not run.
#
# A compute node starts a fresh non-login shell. It does not inherit whatever
# `module load` / `conda activate` you ran to make the precheck work on the login
# node, so the templates used to reach `python -m eeg_pipeline.cli` with no
# environment at all and fail in seconds -- with the reason buried in a per-task
# .err file nobody reads until the whole array is gone.
#
# Set EEG_ENV_SETUP to whatever activates your environment, and submit_validation.sh
# will carry it to every task:
#
#   export EEG_ENV_SETUP='module load miniconda && conda activate eeg-pipeline'
#   export EEG_ENV_SETUP='source ~/venvs/eeg/bin/activate'
#
# sbatch --export=ALL does propagate PATH, so an already-active venv often works
# with EEG_ENV_SETUP unset. Conda usually does not: `conda activate` depends on a
# shell function that a non-interactive shell never defines.

if [[ -n "${EEG_ENV_SETUP:-}" ]]; then
    echo "env setup: ${EEG_ENV_SETUP}"
    # Deliberately unquoted expansion: the value is a shell snippet to run, and
    # it comes from the submitting user's own environment, not from the dataset.
    eval "${EEG_ENV_SETUP}"
fi

# Preflight. Import the two things every task needs BEFORE doing any work, so a
# broken environment reports itself as a broken environment instead of as an
# opaque exit 1 several minutes into a run.
if ! PREFLIGHT="$(python -c '
import sys
print(sys.executable)
import mne, eeg_pipeline
print(mne.__version__)
print(eeg_pipeline.__file__)
' 2>&1)"; then
    {
        echo "PREFLIGHT FAILED on $(hostname): the Python environment is not usable here."
        echo
        echo "$PREFLIGHT"
        echo
        echo "  python on PATH: $(command -v python || echo '<none>')"
        echo "  cwd:            $(pwd)"
        echo
        echo "The precheck runs on the LOGIN node and the array runs on COMPUTE nodes;"
        echo "they do not share shell state. Export EEG_ENV_SETUP with whatever activates"
        echo "your environment and resubmit, e.g."
        echo "  export EEG_ENV_SETUP='module load miniconda && conda activate eeg-pipeline'"
        echo
        echo "If the import error names eeg_pipeline rather than mne, the environment is"
        echo "fine and the repo is simply not importable from cwd -- submit from the repo"
        echo "root, or pip install -e . into the environment."
    } >&2
    exit 1
fi
echo "preflight OK:"
sed 's/^/  /' <<<"$PREFLIGHT"
