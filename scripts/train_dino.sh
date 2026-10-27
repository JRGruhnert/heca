#!/usr/bin/env bash
# Visual RL training with DINO keypoints on global cards 2 and 3:
# Molmo on local card 0 (global 2), DINO on local card 1 (global 3).
# The mirror image is train_roma.sh, which owns cards 0 and 1.
#
#   ./scripts/train_dino.sh              # the plan in c02_train_seq.RUNS
#   PLAN_SHARD=1/3 ./scripts/train_dino.sh
#
# The keypoint backend must match how the scene was fitted: DINO scenes carry
# tapas_img.pt and conditions-vis.joblib.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export CUDA_VISIBLE_DEVICES="2,3"
export HECA_MOLMO_DEVICES=0
export HECA_KP_DEVICE=cuda:1
export HECA_KP_BACKEND=dino
exec python -m scripts.c02_train_seq "$@"
