#!/usr/bin/env bash
# Fit a scene's experts with one arm per GPU; every arm runs the whole pipeline
# (fit -> conditions -> eval) over its share of the experts, sequentially.
#
#   ./scripts/run_experts_gpu.sh [scene] [gpus] [threads] [condition_demos] [episodes]
#
# condition_demos 0 = every demo in the expert's demos.h5.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

py="${PYTHON:-python}"
scene="${1:-scene0}"
gpus="${2:-4}"
threads="${3:-6}"
demos="${4:-0}"
episodes="${5:-100}"

mkdir -p logs/experts

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

mapfile -t tags < <("$py" -c 'import sys
from scripts.common.scenes import experts_for_scene
print("\n".join(cfg.tag for cfg in experts_for_scene(sys.argv[1])))' "$scene")

echo "=== $scene: ${#tags[@]} experts, $gpus GPU(s), $threads threads each"
echo "=== condition demos $demos, eval episodes $episodes"

pids=()
for i in "${!tags[@]}"; do
    tag="${tags[$i]}"
    gpu=$((i % gpus))
    echo "gpu $gpu <- $tag"
    CUDA_VISIBLE_DEVICES="$gpu" "$py" scripts/b01_fit_experts.py \
        --scene "$scene" --model "$tag" --jobs 1 --threads "$threads" \
        --condition-demos "$demos" --episodes "$episodes" \
        > "logs/experts/$tag.log" 2>&1 &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid" || echo "arm $pid exited non-zero" >&2
done

echo "=== done; logs in logs/experts/"
