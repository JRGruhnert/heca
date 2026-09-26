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
log="${HECA_ARM_LOG:-$HOME/arm$index.log}"

# GPU placement. Unset: inherit whatever the launcher shell had (all cards visible).
#   HECA_ARM_CPU=1     -> no GPU at all (the simulator dominates; frees VRAM entirely)
#   HECA_ARM_GPUS=N    -> this shard sees only card (index % N), which stops every
#                         process from paying a CUDA context on all four cards
if [[ -n "${HECA_ARM_CPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=""
elif [[ -n "${HECA_ARM_GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$((index % HECA_ARM_GPUS))"
fi

# Per-arm wandb directories. Prefer the big shared filesystem when the machine has
# one (/export on pearl2), otherwise fall back to $HOME, so the same script works on
# either node without extra setup.
default_wandb_root() {
    for base in "/export/$USER/heca" "$HOME/heca"; do
        if [[ -d "$base" && -w "$base" ]]; then echo "$base/wandb"; return; fi
    done
    echo "$HOME/heca/wandb"
}
wandb_root="${HECA_WANDB_ROOT:-$(default_wandb_root)}"

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

if ! mkdir -p "$WANDB_DIR" 2>/dev/null; then
    echo "error: cannot create WANDB_DIR=$WANDB_DIR" >&2
    echo "       set HECA_WANDB_ROOT to a writable path on this machine" >&2
    exit 1
fi
if [[ ! -e "$(dirname "$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)")" ]]; then
    echo "error: repository parent directory does not exist" >&2
    exit 1
fi

{
    echo "=== $(date '+%F %T') | shard $index/$total | module $module"
    echo "=== python=$python_bin"
    echo "=== WANDB_DIR=$WANDB_DIR log=$log"
    if [[ -f .env ]]; then
        echo "=== .env: $(grep -E '^(HECA_DATA|WANDB_DIR|WANDB_MODE|MUJOCO_GL)=' .env | tr '\n' ' ')"
    else
        echo "=== WARNING: no .env in $(pwd) — HECA_DATA falls back to <repo>/data"
    fi
    # the root the run will actually read: catches a wrong/missing data dir (and a
    # broken import) before 64 processes are started
    data_root_line="$(python -c 'from heca.misc.paths import data_root; print(data_root())' 2>&1)"
    echo "=== data root: $data_root_line"
    if [[ -d "$data_root_line" ]]; then
        echo "=== experts there: $(find "$data_root_line" -name 'tapas_gt.pt' 2>/dev/null | wc -l)"
    fi
} | tee -a "$log"

if [[ -n "${HECA_ARM_DRY:-}" ]]; then
    echo "=== dry run: environment is fine, not starting training"
    exit 0
fi

python -m "$module" 2>&1 | tee -a "$log"
