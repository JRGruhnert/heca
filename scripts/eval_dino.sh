#!/usr/bin/env bash
# Evaluate the fitted DINO pass only: no policy fit, no condition fit.
#
#   ./scripts/eval_dino.sh [scene] [threads] [episodes]
#
# Loads tapas_img.pt + conditions-vis.joblib per expert and runs the stage-3
# evaluation (100 episodes per expert by default). Writes
# plots/eval_success_<scene>.json|png.
#
# Cards as in fit_dino.sh: global 2 and 3, Molmo local 0 (unused here), DINO
# local 1. Requires the fit_dino.sh pass to have finished.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export HECA_EXPERTS_GROUPS="2,3"
export HECA_EXPERTS_MOLMO_LOCAL=0
export HECA_EXPERTS_KP_LOCAL=1
export HECA_KP_BACKEND=dino
export HECA_EXPERTS_ARGS="--skip-fit --skip-conditions"
exec ./scripts/run_experts_gpu.sh "${1:-scene0}" 2 "${2:-8}" 0 "${3:-100}"
