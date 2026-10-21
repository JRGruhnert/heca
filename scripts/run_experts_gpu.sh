#!/usr/bin/env bash
#
# Fit a scene's experts with one arm per GPU. Every arm takes its share of the
# experts and runs the whole pipeline on each of them sequentially:
# fit (tapas_img.pt) -> conditions (conditions-vis.joblib) -> evaluation.
#
#   ./scripts/run_experts_gpu.sh                        # scene0, 4 GPUs, all demos, 100 episodes
#   ./scripts/run_experts_gpu.sh scene0 4 6 20 10       # scene, gpus, threads, condition demos, episodes
#
# condition demos 0 = every demo in the expert's demos.h5 (450 for scene0), which is
# what the conditions are built from when no cap is given.
#
# Before the arms start, the slide ends the visual joint is projected onto are
# derived once and saved into the scene: every arm would otherwise derive its own
# from its own pool, which is 30+ minutes each and gives slightly different ends
# per expert.
#
# GPU memory: during the fit an arm needs ~1.5 GB, but the conditions stage loads
# Molmo (~19 GB in float32) on the card it was given. One arm per GPU assumes each
# card has that much free; check with nvidia-smi first.
#
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

scene="${1:-scene0}"
gpus="${2:-4}"
threads="${3:-6}"
condition_demos="${4:-0}"
episodes="${5:-100}"

mkdir -p logs/experts

# the encoders need the clicked references: one annotation.json + pos.png per label
missing=$(
    for label in $(python -c "
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config
print(' '.join(Scene.get(find_scene_config('$scene'), auto_load=False).entities))
"); do
        [[ -f "data/scenes/ogbench/$scene/$label/annotation.json" ]] || echo "$label"
    done
)
if [[ -n "$missing" ]]; then
    echo "error: $scene has no clicked references for: $missing" >&2
    echo "       copy data/scenes/ogbench/$scene/<label>/{annotation.json,pos.png}" >&2
    echo "       or run scripts/d01_select_references.py --scene $scene" >&2
    exit 1
fi

demos_text="$condition_demos"
[[ "$condition_demos" -eq 0 ]] && demos_text="all"

echo "=== scene $scene | $gpus arm(s) | $threads threads each"
echo "=== condition demos $demos_text | eval episodes $episodes"

# --- one global derivation of the slide ends -----------------------------------
python - "$scene" <<'PY'
import sys

from heca.experts.expert import ExpertModel
from scripts.common.scenes import find_scene_models

cfg = find_scene_models(sys.argv[1])[0]
model = ExpertModel.get(cfg, auto_load=False)
model.use_gt(False)
model.load_extras(derive_extras=True)
model.scene.finish_extras()
model.scene.save()
print("slide ends derived and saved:",
      {label: {k: [round(float(v), 4) for v in p] for k, p in refs.items()}
       for label, refs in model.scene.extra_references.items()})
PY

# --- the experts of the scene, in config order --------------------------------
mapfile -t tags < <(python - "$scene" <<'PY'
import sys

from scripts.common.scenes import experts_for_scene

for cfg in experts_for_scene(sys.argv[1]):
    print(cfg.tag)
PY
)
echo "=== ${#tags[@]} expert(s): ${tags[*]}"

# --- one arm per GPU, round robin over the experts ----------------------------
pids=()
for ((i = 0; i < ${#tags[@]}; i++)); do
    gpu=$((i % gpus))
    tag="${tags[$i]}"
    echo "gpu $gpu <- $tag"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/b01_fit_experts.py \
        --scene "$scene" --model "$tag" --jobs 1 --threads "$threads" \
        --condition-demos "$condition_demos" --episodes "$episodes" \
        > "logs/experts/$tag.log" 2>&1 &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

echo "=== done (exit $status); per-expert logs in logs/experts/"
echo "=== policies found: $(find data/scenes -name 'tapas_img.pt' | wc -l)"
