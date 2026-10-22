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

mkdir -p logs/experts

if [[ -n "${HECA_EXPERTS_DRY:-}" ]]; then
    # no preflight: deriving the slide ends reads every demos.h5 and encodes demo
    # frames, so a dry run must not need the data or the GPU
    mapfile -t tags < <("$py" -c 'import sys
from scripts.common.scenes import experts_for_scene
for cfg in experts_for_scene(sys.argv[1]):
    print("TAG " + cfg.tag)' "$scene" | sed -n 's/^TAG //p')
    echo "=== $scene: ${#tags[@]} experts, $gpus process(es), extra args '${extra}'"
    for i in "${!tags[@]}"; do
        echo "would run: gpu $((i % gpus)) <- ${tags[$i]} $extra"
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
    # one save dir per expert; the conditions live beside its checkpoint
    "$py" -c 'import shutil, sys
from heca.experts.expert import ExpertModel
from scripts.common.scenes import experts_for_scene
for cfg in experts_for_scene(sys.argv[1]):
    for name in ("conditions-vis.joblib", "conditions-gt.joblib"):
        src = ExpertModel.save_dir(cfg) / name
        dst = src.parent / (src.stem + ".bak.joblib")
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"backed up {dst.relative_to(ExpertModel.root)}")' "$scene"
fi

echo "=== $scene: ${#tags[@]} experts, $gpus process(es), $threads threads each"
echo "=== condition demos $demos, eval episodes $episodes, extra args '${extra}'"

run_one() {
    local tag="$1" gpu="$2"
    echo "gpu $gpu <- $tag"
    # shellcheck disable=SC2086  # extra is a deliberately split argument list
    CUDA_VISIBLE_DEVICES="$gpu" "$py" scripts/b01_fit_experts.py \
        --scene "$scene" --model "$tag" --jobs 1 --threads "$threads" \
        --condition-demos "$demos" --episodes "$episodes" $extra \
        > "logs/experts/$tag.log" 2>&1
}

pids=()
for i in "${!tags[@]}"; do
    tag="${tags[$i]}"
    gpu=$((i % gpus))
    if [[ -n "${HECA_EXPERTS_DRY:-}" ]]; then
        echo "would run: gpu $gpu <- $tag $extra"
        continue
    fi
    run_one "$tag" "$gpu" &
    pids+=($!)
    # one process per card: a second Molmo on the same card does not fit
    if (( ${#pids[@]} >= gpus )); then
        wait -n || echo "a process exited non-zero" >&2
        pids=("${pids[@]:1}")
    fi
done

for pid in "${pids[@]}"; do
    wait "$pid" || echo "arm $pid exited non-zero" >&2
done

echo "=== done; logs in logs/experts/"
