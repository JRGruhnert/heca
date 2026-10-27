#!/usr/bin/env bash
# Full visual pass with RoMa v2 keypoints: fit the TAPAS policies, build the
# conditions, evaluate.
#
#   ./scripts/fit_roma.sh [scene] [threads] [episodes]
#
# Global cards 0 and 1: Molmo on local card 0 (global 0), RoMa on local card 1
# (global 1). One process walks the ten experts of the scene, so it owns those two
# cards -- run fit_dino.sh beside it, which owns cards 2 and 3.
#
# Artifacts per expert: tapas_img-roma.pt, conditions-vis-roma.joblib,
# plots/eval_success_<scene>-roma.json|png. Existing files are kept as *.bak first.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export HECA_EXPERTS_GROUPS="0,1"
export HECA_EXPERTS_MOLMO_LOCAL=0
export HECA_EXPERTS_KP_LOCAL=1
export HECA_KP_BACKEND=roma
export HECA_EXPERTS_BACKUP=1
exec ./scripts/run_experts_gpu.sh "${1:-scene0}" 2 "${2:-8}" 0 "${3:-100}"
