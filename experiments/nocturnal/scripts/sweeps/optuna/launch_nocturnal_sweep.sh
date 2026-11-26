#!/usr/bin/env bash
# Launch a nocturnal Optuna sweep worker for one model family.
#
# Resolves the correct model-family interpreter (.venvs/<family>) and runs that
# family's study against the shared nocturnal journal DB. Run it once per worker
# (e.g. in separate tmux panes / GPUs) — all workers for a family cooperate on
# the same study, and different families are separate studies in the same DB.
#
# Usage:
#   experiments/nocturnal/scripts/sweeps/optuna/launch_nocturnal_sweep.sh <family> [n_trials] [gpu]
#
# Examples:
#   .../launch_nocturnal_sweep.sh naive_baseline 50
#   .../launch_nocturnal_sweep.sh deepar 200 0
#   CONFIG=path/to/overrides.yaml .../launch_nocturnal_sweep.sh chronos2 200 1

set -euo pipefail

FAMILY="${1:?Usage: launch_nocturnal_sweep.sh <family> [n_trials] [gpu]}"
N_TRIALS="${2:-50}"
GPU="${3:-}"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "Error: run from within the repository"; exit 1;
}
cd "${REPO_ROOT}"

# Family -> its model-family venv (AutoGluon families share .venvs/autogluon;
# tsmixer -> .venvs/darts; moirai/moirai2 -> .venvs/moirai; timesfm -> .venvs/timesfm).
case "${FAMILY}" in
    chronos2|tide|deepar|patchtst|tft|naive_baseline|statistical|gbm) VENV="autogluon" ;;
    moirai|moirai2) VENV="moirai" ;;
    timesfm) VENV="timesfm" ;;
    tsmixer) VENV="darts" ;;
    *) VENV="${FAMILY}" ;;
esac

PYTHON="${REPO_ROOT}/.venvs/${VENV}/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "Error: ${PYTHON} not found. Create it with:"
    echo "  source scripts/setup_model_env.sh ${VENV}"
    exit 1
fi

CONFIG_ARGS=()
[ -n "${CONFIG:-}" ] && CONFIG_ARGS+=(--config "${CONFIG}")
GPU_ARGS=()
[ -n "${GPU}" ] && GPU_ARGS+=(--gpu-label "${GPU}")

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
[ -n "${GPU}" ] && export CUDA_VISIBLE_DEVICES="${GPU}"

echo "Launching nocturnal '${FAMILY}' sweep (venv=${VENV}, n_trials=${N_TRIALS}, gpu=${GPU:-cpu})"
# Ensure the study exists, then run a worker.
"${PYTHON}" -m "experiments.nocturnal.hpo.${FAMILY}.study" create "${CONFIG_ARGS[@]}"
exec "${PYTHON}" -m "experiments.nocturnal.hpo.${FAMILY}.study" run \
    --n-trials "${N_TRIALS}" "${CONFIG_ARGS[@]}" "${GPU_ARGS[@]}"
