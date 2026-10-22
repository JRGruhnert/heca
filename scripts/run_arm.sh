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
#   HECA_ARM_CORES    CPU budget for this shard (default: uncapped)
#   HECA_ARM_THREADS  BLAS threads               (default: HECA_ARM_CORES, else unset)
#   HECA_ARM_GPUS     pin to card index%N        (default: the card with most free)
#   HECA_ARM_KP_GPU    share card N for all keypoint encoders, Molmo on i%N
#   HECA_ARM_CPU=1    no GPU at all
#
set -euo pipefail

usage() { sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[[ $# -ge 2 ]] || usage

index="$1"
total="$2"
module="${3:-scripts.c02_train_seq_fl}"

env_name="${HECA_CONDA_ENV:-hecarim}"
# the log follows the session prefix, so a second family of arms (e.g.
# HECA_ARM_PREFIX=seq for the non-federated runs) does not append to the first one
prefix="${HECA_ARM_PREFIX:-arm}"
log="${HECA_ARM_LOG:-$HOME/$prefix$index.log}"
# read before any OMP export: nproc answers with OMP_NUM_THREADS once that is set
available="$(nproc)"

# CPU. HECA_ARM_CORES=N means "this shard gets N cores worth of CPU" -- a budget,
# never a core list: cores on this node are shared with runs started by other
# people, so nothing is pinned to specific ones. The budget is enforced as a cgroup
# CPU quota when a user systemd manager is reachable, and otherwise only as thread
# caps (the log says which). Unset: no cap at all.
#   HECA_ARM_CORES=20  -> CPUQuota=2000% and OMP/MKL/OPENBLAS/NUMEXPR=20
cores="${HECA_ARM_CORES:-}"

# BLAS/OpenMP threads per process, for the single-process scripts that do not pin
# them themselves (the federated ones do it per rank from --threads). These are what
# actually bounds a CPU op; without them a single torch call still asks for 64.
threads="${HECA_ARM_THREADS:-}"
if [[ -z "$threads" && -n "$cores" ]]; then
    threads="$cores"
fi
if [[ -n "$threads" ]]; then
    export OMP_NUM_THREADS="$threads"
    export MKL_NUM_THREADS="$threads"
    export OPENBLAS_NUM_THREADS="$threads"
    export NUMEXPR_NUM_THREADS="$threads"
fi

# GPU placement. Unset: inherit whatever the launcher shell had (all cards visible),
# and heca's import-time pick then takes the card with the most free memory -- so
# several arms started together spread themselves over the free cards with nothing
# to configure, and a half-full node still lands where there is room.
#   HECA_ARM_CPU=1     -> no GPU at all (the simulator dominates; frees VRAM entirely)
#   HECA_ARM_GPUS=N    -> this shard sees only card (index % N), which stops every
#                         process from paying a CUDA context on all four cards
#   HECA_ARM_KP_GPU=N -> the keypoint encoders of every shard share card N, and
#                        shard i puts its Molmo on card (i % N). One process then
#                        sees two cards: its own Molmo card, and N with the
#                        keypoint models of all shards. Three arms on four cards,
#                        nothing spilling to host.
if [[ -n "${HECA_ARM_KP_GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$((index % HECA_ARM_KP_GPU)),${HECA_ARM_KP_GPU}"
    export HECA_MOLMO_DEVICES=0      # visible index of this shard's Molmo card
    export HECA_KP_DEVICE=cuda:1     # visible index of the shared keypoint card
elif [[ -n "${HECA_ARM_CPU:-}" ]]; then
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

# How many cards this process can actually see. A CPU fall back trains for days and
# looks exactly like a run, so it is checked here instead of being inferred later
# from "Running on cpu" somewhere in the log.
gpus="none requested (HECA_ARM_CPU)"
if [[ -z "${HECA_ARM_CPU:-}" ]]; then
    gpus="$(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>&1)"
fi

# A cgroup quota is the only way to cap a process without naming cores. Probe it
# once, so the log states whether the budget is enforced or only advised.
cpu_guard=()
quota="unlimited"
if [[ -n "$cores" ]]; then
    pct=$(( cores * 100 ))
    if systemd-run --user --scope --quiet -p "CPUQuota=${pct}%" true >/dev/null 2>&1; then
        cpu_guard=(systemd-run --user --scope --quiet -p "CPUQuota=${pct}%")
        quota="CPUQuota=${pct}%"
    else
        quota="threads only (no user systemd manager for a CPUQuota)"
    fi
fi

{
    echo "=== $(date '+%F %T') | shard $index/$total | module $module"
    echo "=== python=$python_bin"
    echo "=== WANDB_DIR=$WANDB_DIR log=$log"
    echo "=== CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES-unset}' gpus=$gpus"
    if [[ -n "${HECA_KP_DEVICE:-}" ]]; then
        echo "=== keypoint encoder on ${HECA_KP_DEVICE}," \
            "Molmo on visible card ${HECA_MOLMO_DEVICES:-all}"
    fi
    echo "=== cpus=$quota of $available, threads=$threads"
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

if [[ -z "${HECA_ARM_CPU:-}" ]]; then
    if ! [[ "$gpus" =~ ^[0-9]+$ ]]; then
        echo "error: cannot query CUDA: $gpus" >&2
        exit 1
    fi
    if (( gpus == 0 )); then
        echo "error: no CUDA card visible to this shard" \
            "(CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES-unset}')." >&2
        echo "       Refusing to train on the CPU. Unset CUDA_VISIBLE_DEVICES /" >&2
        echo "       HECA_ARM_GPUS, or set HECA_ARM_CPU=1 for a CPU run on purpose." >&2
        exit 1
    fi
fi

if [[ -n "${HECA_ARM_DRY:-}" ]]; then
    echo "=== dry run: environment is fine, not starting training"
    exit 0
fi

if (( ${#cpu_guard[@]} )); then
    "${cpu_guard[@]}" python -m "$module" 2>&1 | tee -a "$log"
else
    python -m "$module" 2>&1 | tee -a "$log"
fi
