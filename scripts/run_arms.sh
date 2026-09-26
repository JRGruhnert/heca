#!/usr/bin/env bash
#
# Launch training shards in detached tmux sessions, one session per shard:
#
#   ./scripts/run_arms.sh 0 4 8      # shards 0..3 of 8  (start with this)
#   ./scripts/run_arms.sh 4 4 8      # shards 4..7 of 8  (add later, same total!)
#
# <first> <count> <total> — the shard count is the *final* partition, not the
# number of jobs you start now. Splitting the table with 4 and later with 8 would
# overlap: PLAN_SHARD=0/4 already owns all runs, so the extra jobs from a /8 split
# would duplicate work and write into the same checkpoint directories. Decide the
# total once, launch it in waves.
#
# Overridable via environment:
#   HECA_ARM_STAGGER  seconds between session starts (default: 15)
#   HECA_ARM_PREFIX   tmux session name prefix     (default: arm)
#   HECA_ARM_MODULE   training module to run       (default: scripts.c02_train_seq_fl)
#
set -euo pipefail

usage() { sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[[ $# -ge 3 ]] || usage

first="$1"
count="$2"
total="$3"
stagger="${HECA_ARM_STAGGER:-15}"
prefix="${HECA_ARM_PREFIX:-arm}"
module="${HECA_ARM_MODULE:-scripts.c02_train_seq_fl}"

worker="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)/run_arm.sh"
[[ -x "$worker" ]] || { echo "error: $worker is not executable" >&2; exit 1; }
(( first >= 0 && count > 0 && first + count <= total )) || {
    echo "error: need 0 <= first, count > 0 and first+count <= total" >&2
    exit 1
}

for (( i = first; i < first + count; i++ )); do
    name="$prefix$i"
    if tmux has-session -t "$name" 2>/dev/null; then
        echo "skip: session $name already exists"
        continue
    fi
    # tmux panes inherit the *server's* environment, not this shell's, so the
    # placement variables have to travel inside the command string.
    placement="HECA_ARM_CPU='${HECA_ARM_CPU:-}' HECA_ARM_GPUS='${HECA_ARM_GPUS:-}'"
    tmux new -d -s "$name" "$placement $worker $i $total $module"
    echo "started $name -> shard $i/$total  [$placement]"
    sleep "$stagger"
done

cat <<'EOF'

check after a few seconds:
  tmux ls                                   # one session per shard
  head -3 ~/arm0.log                        # "shard 0/8 | python=…/envs/hecarim/bin/python"
  grep -m1 "run(s) planned" ~/arm0.log      # must say "(shard 0/8)" — if it says 48, kill it
EOF
