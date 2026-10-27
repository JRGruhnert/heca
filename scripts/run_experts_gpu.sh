#!/usr/bin/env bash
# Fit a scene's experts with one process per GPU; every process runs the whole
# pipeline (fit -> conditions -> eval) over its share of the experts, sequentially.
#
#   ./scripts/run_experts_gpu.sh [scene] [gpus] [threads] [condition_demos] [episodes]
#
# condition_demos 0 = every demo in the expert's demos.h5.
#
# Overridable via environment:
#   PYTHON              interpreter to use            (default: python)
#   HECA_EXPERTS_ARGS   extra args for b01_fit_experts, e.g. "--skip-fit --skip-eval"
#                       to rebuild only the conditions
#   HECA_EXPERTS_CARDS_PER_PROC  cards each process may use (default: 1). A Molmo
#                       that does not fit one card is spread over two; then only
#                       gpus/2 processes run at once, each on its own pair.
#   HECA_EXPERTS_GROUPS one group of GLOBAL card indices per process, e.g.
#                       "0,2 1,2" for two processes over three cards (groups may
#                       overlap). Setting it overrides the gpus / CARDS_PER_PROC
#                       arguments and runs one process per group.
#   HECA_EXPERTS_MOLMO_LOCAL  LOCAL index inside a group that carries the Molmo
#   HECA_EXPERTS_KP_LOCAL     LOCAL index inside a group that carries the keypoint
#                       encoder, i.e. the card the process sees as cuda:0, cuda:1.
#                       They are exported as HECA_MOLMO_DEVICES / HECA_KP_DEVICE
#                       -- plain local indices, the same spelling for both --
#                       while CUDA_VISIBLE_DEVICES holds the group itself, so
#                       "GROUPS=0,1 MOLMO_LOCAL=0 KP_LOCAL=1" puts the Molmo on
#                       global card 0 and the DINO / RoMa model on global card 1.
#   HECA_EXPERTS_MOLMO_GPUS / HECA_EXPERTS_KP_GPU  shorthand for exactly that
#                       layout: one process per listed Molmo card, every process
#                       sharing the single keypoint card.
#   HECA_EXPERTS_DRY    print the commands instead of running them
#   HECA_EXPERTS_BACKUP copy each conditions file to <stem>.bak.joblib when the
#                       target does not exist yet, so a rebuild stays reversible
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

py="${PYTHON:-python}"
scene="${1:-scene0}"
gpus="${2:-4}"
threads="${3:-6}"
demos="${4:-0}"
episodes="${5:-100}"
extra="${HECA_EXPERTS_ARGS:-}"
[[ -n "${HECA_KP_BACKEND:-}" ]] && extra="$extra --kp-backend $HECA_KP_BACKEND"
# cards per process: a Molmo that does not fit one card is spread over several, and
# then fewer processes run at once (one per group of cards)
per="${HECA_EXPERTS_CARDS_PER_PROC:-1}"
# Layout "one Molmo per card, all keypoint encoders on one card":
#   HECA_EXPERTS_MOLMO_GPUS=0,1  HECA_EXPERTS_KP_GPU=2
# Worker i then sees (molmo[i], kp) with its Molmo on visible card 0 and its
# keypoint encoder on visible card 1, so the two Molmos never share a card and the
# DINO / RoMa models of all workers fit on the third one.
molmo_gpus="${HECA_EXPERTS_MOLMO_GPUS:-}"
kp_gpu="${HECA_EXPERTS_KP_GPU:-}"

# General form: one group of GLOBAL card indices per worker, plus the LOCAL index
# inside a group that carries each model.
#   HECA_EXPERTS_GROUPS="0,2 1,2" HECA_EXPERTS_MOLMO_LOCAL=0 HECA_EXPERTS_KP_LOCAL=1
# means: worker A sees global cards 0 and 2, worker B sees 1 and 2 (the keypoint card
# is shared), and inside each worker the Molmo is local card 0 while the keypoint
# encoder is local card 1. Groups may overlap; they need not be disjoint.
groups_str="${HECA_EXPERTS_GROUPS:-}"
molmo_local="${HECA_EXPERTS_MOLMO_LOCAL:-}"
kp_local="${HECA_EXPERTS_KP_LOCAL:-}"
if [[ -z "$groups_str" && -n "$molmo_gpus" && -n "$kp_gpu" ]]; then
    # older shorthand: one Molmo per listed card, all keypoint encoders on one card
    groups_str=""
    for card in ${molmo_gpus//,/ }; do
        groups_str+="${card},${kp_gpu} "
    done
    molmo_local=0
    kp_local=1
fi
groups=()
if [[ -n "$groups_str" ]]; then
    read -r -a groups <<< "$groups_str"
fi
if (( ${#groups[@]} )); then
    slots=${#groups[@]}
    per=1
elif (( per < 1 || gpus % per != 0 )); then
    echo "error: $gpus GPU(s) cannot be split into groups of $per" >&2
    exit 1
else
    slots=$(( gpus / per ))
fi

# the card group of run i modulo the number of slots
cards_for() {
    local k=$1 out=""
    for (( c = 0; c < per; c++ )); do
        out+="${out:+,}$(( k * per + c ))"
    done
    echo "$out"
}


mkdir -p logs/experts

if [[ -n "${HECA_EXPERTS_DRY:-}" ]]; then
    # no preflight: deriving the slide ends reads every demos.h5 and encodes demo
    # frames, so a dry run must not need the data or the GPU
    mapfile -t tags < <("$py" -c 'import sys
from scripts.common.scenes import experts_for_scene
for cfg in experts_for_scene(sys.argv[1]):
    print("TAG " + cfg.tag)' "$scene" | sed -n 's/^TAG //p')
    echo "=== $scene: ${#tags[@]} experts, $slots process(es) of $per GPU, extra args '${extra}'"
    for i in "${!tags[@]}"; do
        if (( ${#groups[@]} )); then
            echo "would run: gpus ${groups[$(( i % ${#groups[@]} ))]} (molmo local ${molmo_local:-?}, kp local ${kp_local:-?}) <- ${tags[$i]} $extra"
        else
            echo "would run: gpus $(cards_for $(( i % slots ))) <- ${tags[$i]} $extra"
        fi
    done
    exit 0
fi

# the slide ends the visual joint is projected onto: derived and saved once, so the
# arms share them instead of each deriving its own pool
"$py" -c 'import sys
from heca.experts.expert import ExpertModel
from scripts.common.scenes import find_scene_models
model = ExpertModel.get(find_scene_models(sys.argv[1])[0], auto_load=False)
model.use_gt(False)
model.load_extras(True)
model.scene.finish_extras()
model.scene.save()
print("slide ends:", {k: list(v) for k, v in model.scene.extra_references.items()})' "$scene"

# the marker survives the logger's own stdout output, which would otherwise land in
# this list as if it were a tag
mapfile -t tags < <("$py" -c 'import sys
from scripts.common.scenes import experts_for_scene
for cfg in experts_for_scene(sys.argv[1]):
    print("TAG " + cfg.tag)' "$scene" | sed -n 's/^TAG //p')

if [[ -n "${HECA_EXPERTS_BACKUP:-}" ]]; then
    # one save dir per expert: conditions and fitted policies live beside it
    "$py" -c 'import shutil, sys
from heca.experts.expert import ExpertModel
from scripts.common.scenes import experts_for_scene
for cfg in experts_for_scene(sys.argv[1]):
    save_dir = ExpertModel.save_dir(cfg)
    files = [save_dir / n for n in ("conditions-vis.joblib", "conditions-gt.joblib")]
    files += sorted(save_dir.glob("tapas_*.pt"))  # a refit costs hours
    for src in files:
        dst = src.with_suffix(".bak" + src.suffix)
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"backed up {dst.relative_to(ExpertModel.root)}")' "$scene"
fi

echo "=== $scene: ${#tags[@]} experts, $slots process(es) of $per GPU, $threads threads"
echo "=== condition demos $demos, eval episodes $episodes, extra args '${extra}'"

run_one() {
    local tag="$1" cards="$2"
    echo "gpus $cards <- $tag"
    # shellcheck disable=SC2086  # extra is a deliberately split argument list
    CUDA_VISIBLE_DEVICES="$cards" "$py" scripts/b01_fit_experts.py \
        --scene "$scene" --model "$tag" --jobs 1 --threads "$threads" \
        --condition-demos "$demos" --episodes "$episodes" $extra \
        > "logs/experts/$tag.log" 2>&1
}

pids=()
if (( ${#groups[@]} )); then
    # both are local indices into the group, spelled as plain numbers
    [[ -n "$molmo_local" ]] && export HECA_MOLMO_DEVICES="$molmo_local"
    [[ -n "$kp_local" ]] && export HECA_KP_DEVICE="$kp_local"
fi

for i in "${!tags[@]}"; do
    tag="${tags[$i]}"
    if (( ${#groups[@]} )); then
        cards="${groups[$(( i % ${#groups[@]} ))]}"
    else
        cards="$(cards_for $(( i % slots )))"
    fi
    run_one "$tag" "$cards" &
    pids+=($!)
    # one process per card group: a second Molmo beside the first does not fit
    if (( ${#pids[@]} >= slots )); then
        wait -n || echo "a process exited non-zero" >&2
        pids=("${pids[@]:1}")
    fi
done

for pid in "${pids[@]}"; do
    wait "$pid" || echo "arm $pid exited non-zero" >&2
done

echo "=== done; logs in logs/experts/"
