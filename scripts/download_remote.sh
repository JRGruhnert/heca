#!/usr/bin/env bash
#
# Pull the run artefacts out of a remote HECA tree over rsync+ssh: the learner
# checkpoints of every training run, the fitted conditions and the plots.
# Nothing else travels -- demo h5 files, reference pictures, eval logs and the
# rest of the data directory stay on the remote.
#
#   ./scripts/download_remote.sh --remote gorilla:~/heca/data
#   ./scripts/download_remote.sh --remote gorilla:~/heca/data --jump me@login
#   ./scripts/download_remote.sh --dry-run --stats     # uses the settings below
#
# Safe to interrupt: an interrupted file resumes from the byte it reached, and
# a re-run moves only what is still missing.
#
# --remote         remote data directory, in rsync's host:path form
# --dest           local data directory to fill                       (default: data)
# --jump           ssh login/bastion host to hop through, e.g. me@login.example
# --password-file  file holding the ssh password, chmod 600 (needs sshpass);
#                  leave it empty and ssh asks, which is the safer option
# --dry-run        report what would be transferred, change nothing
# --list           print the matching remote paths and stop
# --stats          print the transfer statistics
# --with-experts   also pull the fitted expert models (experts/*/*.pt, ~8 GB);
#                  they are not checkpoints of a training run, off by default
# --results        pull the eval JSONs and every image (.png/.jpg), without weights
# --conditions     pull the fitted conditions (condition*.joblib), i.e. the gt,
#                  pos, dino and roma variants, and nothing else
#
# With neither --results nor --conditions the script pulls the whole artefact set
# (checkpoints, conditions and plots); the two selectors combine, so
# ``--results --conditions`` gives the numbers, figures and fitted conditions.
#
set -euo pipefail

usage() {
    awk 'NR > 2 && /^set -euo pipefail/ { exit } NR > 2' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

#: the checkout this script lives in, so DEST does not depend on where it is run
REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"

# --- cluster settings: fill these in once ------------------------------------
REMOTE="gruhneja@10.4.28.57:~/heca/data"                # e.g. me@server:~/heca/data
JUMP="gruhneja@login.informatik.uni-freiburg.de"                    # e.g. me@login.example, empty = direct
PASSWORD_FILE="${HECA_PASSWORD_FILE:-}"  # e.g. ~/.heca_pw, chmod 600
PASSWORD=""                              # or paste it here; written to a 0600
                                         # temp file, so it never shows in ps
DEST="${HECA_DEST:-$REPO_ROOT/data-server}"  # where it lands locally
# ----------------------------------------------------------------------------

remote="$REMOTE"
dest="$DEST"
jump="$JUMP"
password_file="$PASSWORD_FILE"
dry_run=0
list_only=0
stats=0
with_experts=0
results=0
conditions=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote)        remote="$2";        shift 2 ;;
        --dest)          dest="$2";          shift 2 ;;
        --jump)          jump="$2";          shift 2 ;;
        --password-file) password_file="$2"; shift 2 ;;
        --dry-run)       dry_run=1;          shift ;;
        --list)          list_only=1;        shift ;;
        --stats)         stats=1;            shift ;;
        --with-experts)  with_experts=1;     shift ;;
        --results)       results=1;          shift ;;
        --conditions)    conditions=1;       shift ;;
        -h|--help)       usage ;;
        *) echo "error: unknown argument '$1'" >&2; usage ;;
    esac
done

[[ -n "$remote" ]] || {
    echo "error: no remote set; pass --remote or fill in REMOTE in this script" >&2
    usage
}

# a trailing slash means "the contents of", which is what --dest expects
[[ "$remote" == */ ]] || remote="$remote/"

filters="$(mktemp)"
trap 'rm -f "$filters"' EXIT
if [[ $results == 1 || $conditions == 1 ]]; then
    echo '+ */' >>"$filters"
    if [[ $results == 1 ]]; then
        printf '%s\n' '+ *.json' '+ *.png' '+ *.jpg' '+ *.jpeg' >>"$filters"
    fi
    if [[ $conditions == 1 ]]; then
        echo '+ scenes/*/*/experts/*/condition*.joblib' >>"$filters"
    fi
else
    cat >"$filters" <<'EOF'
# descend into every directory, so the patterns below are reachable
+ */
# 1. learner checkpoints of the training runs (network/<folder>/<run tag>/ckp_*.pt)
+ network/**/ckp_*.pt
# 2. fitted conditions (conditions-gt.joblib / condition-gt.joblib / ...)
+ scenes/*/*/experts/*/condition*.joblib
# 3. plots: per expert, per scene, plus the loose scene-level figures
+ scenes/*/*/plots/***
+ scenes/*/*/experts/*/plots/***
+ scenes/*/*/*.png
EOF
    # the fitted expert models are ~8 GB and are not training checkpoints
    [[ $with_experts == 1 ]] && echo '+ scenes/*/*/experts/*/*.pt' >>"$filters"
fi
echo '- *' >>"$filters"

ssh_cmd="ssh"
[[ -n "$jump" ]] && ssh_cmd+=" -J $jump"
[[ -n "${HECA_SSH:-}" ]] && ssh_cmd+=" $HECA_SSH"
if [[ -n "$password_file" ]]; then
    command -v sshpass >/dev/null || {
        echo "error: --password-file needs sshpass (not found on PATH)" >&2
        exit 1
    }
    ssh_cmd="sshpass -f $password_file $ssh_cmd"
elif [[ -n "$PASSWORD" ]]; then
    command -v sshpass >/dev/null || {
        echo "error: PASSWORD needs sshpass (not found on PATH)" >&2
        exit 1
    }
    # not passed as -p: that would put the secret in the process list
    secret="$(mktemp)"
    chmod 600 "$secret"
    printf '%s' "$PASSWORD" >"$secret"
    trap 'rm -f "$filters" "$secret"' EXIT
    ssh_cmd="sshpass -f $secret $ssh_cmd"
fi

# --inplace --append-verify is what makes an interrupted run resume: the partial
# file keeps its destination name, so the next run continues from the byte it
# reached, and the checksum guards against resuming onto a stale prefix.
opts=(-a --prune-empty-dirs --human-readable
      --inplace --append-verify --timeout=60
      --filter="merge $filters"
      -e "$ssh_cmd")
if [[ -n "${HECA_RSYNC:-}" ]]; then
    read -ra extra <<<"${HECA_RSYNC}"
    opts+=("${extra[@]}")
fi
[[ $dry_run   == 1 ]] && opts+=(-n)
[[ $stats     == 1 ]] && opts+=(--stats)
[[ $list_only == 1 ]] && opts+=(--list-only)

echo "remote: $remote"
echo "dest:   $dest/"
[[ -n "$jump" ]] && echo "via:    ssh -J $jump"
if [[ $list_only == 0 ]]; then
    mkdir -p "$dest"
fi

rsync "${opts[@]}" "$remote" "$dest/"

if [[ $list_only == 1 ]]; then
    exit 0
fi
echo
what="checkpoints, conditions and plots"
if [[ $results == 1 || $conditions == 1 ]]; then
    what=""
    [[ $results == 1 ]] && what="the eval JSONs and figures"
    [[ $conditions == 1 ]] && what="${what:+$what, }the fitted conditions"
fi
echo "done: $dest/ now holds $what"
