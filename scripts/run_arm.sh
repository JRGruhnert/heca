#!/usr/bin/env bash
#
# Run one shard of a training script. Meant to be launched by scripts/run_arms.sh
# (which starts each shard in its own tmux session), but it works standalone:
#
#   ./scripts/run_arm.sh <shard index> <shard count> [module]
#   ./scripts/run_arm.sh 3 8                                  # shard 3 of 8
#   ./scripts/run_arm.sh 0 8 scripts.c01_train_fl
#
# It activates the conda environment (inside the pane, because tmux panes inherit
# the *tmux server's* environment, not the shell you started them from), gives the
# shard its own WANDB_DIR, and tees everything to a log file that survives the job.
#
# Overridable via environment:
#   HECA_CONDA_ENV    conda env name             (default: hecarim)
#   CONDA_BASE        conda installation root    (default: autodetected)
#   HECA_WANDB_ROOT   parent of the per-arm dirs (default: /export/$USER/heca/wandb)
#   HECA_ARM_LOG      log file                   (default: $HOME/arm<index>.log)
#
set -euo pipefail

usage() { sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[[ $# -ge 2 ]] || usage

index="$1"
total="$2"
module="${3:-scripts.c02_train_seq_fl}"

env_name="${HECA_CONDA_ENV:-hecarim}"
wandb_root="${HECA_WANDB_ROOT:-/export/$USER/heca/wandb}"
log="${HECA_ARM_LOG:-$HOME/arm$index.log}"

# --- conda: find the base, then activate inside this process -------------------
find_conda_base() {
    if [[ -n "${CONDA_BASE:-}" ]]; then echo "$CONDA_BASE"; return; fi
    if [[ -n "${CONDA_EXE:-}" ]]; then dirname "$(dirname "$CONDA_EXE")"; return; fi
    if command -v conda >/dev/null 2>&1; then conda info --base; return; fi
    for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
        if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then echo "$candidate"; return; fi
    done
    return 1
}
conda_base="$(find_conda_base)" || {
    echo "error: no conda installation found; set CONDA_BASE=/path/to/conda" >&2
    exit 1
}
# shellcheck source=/dev/null
source "$conda_base/etc/profile.d/conda.sh"
conda activate "$env_name"

python_bin="$(command -v python)"
if [[ "$python_bin" != *"/envs/$env_name/"* ]]; then
    echo "error: activated python is $python_bin, not the '$env_name' env" >&2
    exit 1
fi

# --- per-shard environment -----------------------------------------------------
cd "$(dirname "$(readlink -f "$0")")/.."        # repository root
export PLAN_SHARD="$index/$total"
export WANDB_DIR="$wandb_root/job$index"
export PYTHONUNBUFFERED=1

mkdir -p "$WANDB_DIR"
{
    echo "=== $(date '+%F %T') | shard $index/$total | module $module"
    echo "=== python=$python_bin"
    echo "=== WANDB_DIR=$WANDB_DIR log=$log"
} | tee -a "$log"

python -m "$module" 2>&1 | tee -a "$log"
