#!/usr/bin/env bash
# Evaluate the fitted RoMa pass only: no policy fit, no condition fit.
#
#   ./scripts/eval_roma.sh [scene] [threads] [episodes]
#
# Loads tapas_img-roma.pt + conditions-vis-roma.joblib per expert and runs the
# stage-3 evaluation. Writes plots/eval_success_<scene>-roma.json|png.
#
# Cards as in fit_roma.sh: global 0 and 1, Molmo local 0 (unused here), RoMa
# local 1. Requires the fit_roma.sh pass to have finished.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export HECA_EXPERTS_GROUPS="0,1"
export HECA_EXPERTS_MOLMO_LOCAL=0
export HECA_EXPERTS_KP_LOCAL=1
export HECA_KP_BACKEND=roma
export HECA_EXPERTS_ARGS="--skip-fit --skip-conditions"
exec ./scripts/run_experts_gpu.sh "${1:-scene0}" 2 "${2:-8}" 0 "${3:-100}"
