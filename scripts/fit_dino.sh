#!/usr/bin/env bash
# Full visual pass with DINO keypoints: fit the TAPAS policies, build the
# conditions, evaluate.
#
#   ./scripts/fit_dino.sh [scene] [threads] [episodes]
#
# Global cards 2 and 3: Molmo on local card 0 (global 2), DINO on local card 1
# (global 3). One process walks the ten experts of the scene, so it owns those two
# cards -- run fit_roma.sh beside it, which owns cards 0 and 1.
#
# Artifacts per expert: tapas_img.pt, conditions-vis.joblib,
# plots/eval_success_<scene>.json|png. Existing files are kept as *.bak first.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export HECA_EXPERTS_GROUPS="2,3"
export HECA_EXPERTS_MOLMO_LOCAL=0
export HECA_EXPERTS_KP_LOCAL=1
export HECA_KP_BACKEND=dino
export HECA_EXPERTS_BACKUP=1
exec ./scripts/run_experts_gpu.sh "${1:-scene0}" 2 "${2:-8}" 0 "${3:-100}"
