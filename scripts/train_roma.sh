#!/usr/bin/env bash
# Visual RL training with RoMa v2 keypoints on global cards 0 and 1:
# Molmo on local card 0 (global 0), RoMa on local card 1 (global 1).
# The mirror image is train_dino.sh, which owns cards 2 and 3.
#
#   ./scripts/train_roma.sh              # the plan in c02_train_seq.RUNS
#   PLAN_SHARD=0/3 ./scripts/train_roma.sh
#
# The keypoint backend must match how the scene was fitted: RoMa scenes carry
# tapas_img-roma.pt and conditions-vis-roma.joblib, so select them here or the
# arms load the DINO policies by mistake.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export CUDA_VISIBLE_DEVICES="0,1"
export HECA_MOLMO_DEVICES=0
export HECA_KP_DEVICE=cuda:1
export HECA_KP_BACKEND=roma
exec python -m scripts.c02_train_seq "$@"
