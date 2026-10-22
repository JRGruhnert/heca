#!/usr/bin/env bash
# Wait for a conditions refit to finish, check it really produced single-component
# conditions, then launch the training arms.
#
#   ./scripts/train_after_refit.sh [scene] [arms] [cards] [cores] [molmo cap]
#   ./scripts/train_after_refit.sh scene0 3 3 20 8000MiB
#
# The arms load every expert's conditions once, at construction, so starting them
# while the refit is still writing would hand them whichever mix of old (K<=10) and
# new (K=1) files happens to be on disk. The wait and the check are the point of
# this script; the launch at the end is just the two-card-safe configuration.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

py="${PYTHON:-python}"
scene="${1:-scene0}"
arms="${2:-3}"
cards="${3:-3}"
cores="${4:-20}"
cap="${5:-8000MiB}"

echo "=== waiting for the conditions refit to finish ($(date '+%F %T'))"
while pgrep -f 'scripts/b01_fit_experts' >/dev/null; do
    sleep 60
done
echo "=== refit finished ($(date '+%F %T'))"

"$py" - "$scene" <<'PY'
import glob
import sys

import joblib

scene = sys.argv[1]
files = sorted(glob.glob(f"data/scenes/ogbench/{scene}/experts/*/conditions-vis.joblib"))
if not files:
    print(f"error: no conditions-vis.joblib under {scene}", file=sys.stderr)
    raise SystemExit(1)

bad = {}
for path in files:
    pair = joblib.load(path)
    counts = {
        len(pair.pre.models[label].get_parameters()["weights"])
        for label in pair.pre.models
    }
    if counts != {1}:
        bad[path.split("/")[-2]] = sorted(counts)
if bad:
    print(
        f"error: {len(bad)} of {len(files)} experts are not single-component: {bad}",
        file=sys.stderr,
    )
    print("       not launching: the arms would load the old gates", file=sys.stderr)
    raise SystemExit(1)
print(f"=== all {len(files)} experts have one component per entity")
PY

echo "=== launching $arms arm(s) on $cards card(s), Molmo cap $cap, $cores cores each"
HECA_ARM_GPUS="$cards" HECA_MOLMO_GPU_MEMORY="$cap" HECA_ARM_CORES="$cores" \
    HECA_ARM_PREFIX=vis HECA_ARM_MODULE=scripts.c02_train_seq \
    ./scripts/run_arms.sh 0 "$arms" "$arms"
