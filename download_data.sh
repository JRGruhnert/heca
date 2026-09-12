#!/usr/bin/env bash
# Fetch the data bundle produced by ./upload_data.sh and unpack it into ./data,
# so a fresh clone on another machine is ready to run without Git LFS.
#
#   ./download_data.sh                          # use data_bundle.env
#   ./download_data.sh --record 1234567         # from a Zenodo record
#   ./download_data.sh --url https://...zip     # from any direct link
#   ./download_data.sh --remote gdrive:heca-data
#   ./download_data.sh --from-file /media/usb/data.zip
#   ./download_data.sh --list                   # show what is inside
#   ./download_data.sh --force                  # overwrite an existing data/
#
# The archive is verified against the published sha256 before it is unpacked,
# and the .h5 demos / nested .zip archives are intentionally *not* part of it.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# shellcheck source=data_bundle.env
[[ -f data_bundle.env ]] && source ./data_bundle.env

DATA_DIR="${DATA_DIR:-data}"
BUNDLE_ZIP="${BUNDLE_ZIP:-data.zip}"
BUNDLE_BACKEND="${BUNDLE_BACKEND:-zenodo}"
ZENODO_API="${ZENODO_API:-https://zenodo.org/api}"
ZENODO_RECORD="${ZENODO_RECORD:-}"
RCLONE_REMOTE="${RCLONE_REMOTE:-}"
DATA_URL="${DATA_URL:-}"

SOURCE=""            # url | record | remote | file (empty -> data_bundle.env)
URL="$DATA_URL"
RECORD="$ZENODO_RECORD"
REMOTE="$RCLONE_REMOTE"
FROM_FILE=""
EXPECTED_SHA=""
DEST="."
FORCE=0 LIST_ONLY=0 KEEP_ZIP=0 VERIFY_ONLY=0

usage() { sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }
human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 bytes"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)        SOURCE=url;    URL="$2"; shift 2 ;;
    --record)     SOURCE=record; RECORD="$2"; shift 2 ;;
    --remote)     SOURCE=remote; REMOTE="$2"; shift 2 ;;
    --from-file)  SOURCE=file;   FROM_FILE="$2"; shift 2 ;;
    --sha256)     EXPECTED_SHA="$2"; shift 2 ;;
    --dest)       DEST="$2"; shift 2 ;;
    --data-dir)   DATA_DIR="$2"; shift 2 ;;
    --out)        BUNDLE_ZIP="$2"; shift 2 ;;
    --force)      FORCE=1; shift ;;
    --list)       LIST_ONLY=1; shift ;;
    --keep-zip)   KEEP_ZIP=1; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

# The sha256 sits next to the archive, whatever --out says.
BUNDLE_SHA="${BUNDLE_ZIP}.sha256"

target_dir="$DEST/$DATA_DIR"

# --- refuse to clobber a populated tree ------------------------------------
if [[ $FORCE -eq 0 && $LIST_ONLY -eq 0 && $VERIFY_ONLY -eq 0 && -d "$target_dir" ]]; then
  existing=$(find "$target_dir" -type f | head -1 || true)
  if [[ -n "$existing" ]]; then
    die "$target_dir already contains data; rerun with --force to overwrite"
  fi
fi

if [[ $LIST_ONLY -eq 1 ]]; then
  [[ -f "$BUNDLE_ZIP" ]] || die "no local $BUNDLE_ZIP; download it first (without --list)"
  unzip -l "$BUNDLE_ZIP" | head -40
  echo
  info "total entries: $(unzip -l "$BUNDLE_ZIP" | tail -1 | awk '{print $2}')"
  exit 0
fi

# --- resolve the source ----------------------------------------------------
if [[ -z "$SOURCE" ]]; then
  if [[ -n "$FROM_FILE" ]]; then SOURCE=file
  elif [[ "$BUNDLE_BACKEND" == "rclone" && -n "$REMOTE" ]]; then SOURCE=remote
  elif [[ "$BUNDLE_BACKEND" == "file" ]]; then SOURCE=file
  elif [[ -n "$RECORD" ]]; then SOURCE=record
  elif [[ -n "$URL" ]]; then SOURCE=url
  else
    die "nothing configured: set DATA_URL / ZENODO_RECORD in data_bundle.env, or pass --url/--record/--remote/--from-file"
  fi
fi

# Zenodo: turn a record id into the direct file links of the record
zenodo_link() { # $1 = suffix filter (".zip" / ".sha256")
  python3 -c '
import json, sys, urllib.request
api, rid, suffix = sys.argv[1], sys.argv[2], sys.argv[3]
with urllib.request.urlopen(f"{api}/records/{rid}") as fh:
    rec = json.load(fh)
match = [f for f in rec.get("files", []) if f["key"].endswith(suffix)]
print(match[0]["links"]["self"] if match else "")
' "$ZENODO_API" "$1" "$2"
}

fetch_archive() {
  command -v curl >/dev/null || die "curl is required"
  info "downloading $(basename "$BUNDLE_ZIP")"
  # -L follows the redirects Drive/Dropbox/Zenodo use, -C - resumes a partial file
  curl -L --fail --retry 3 --retry-delay 5 -C - -o "$BUNDLE_ZIP" "$URL" \
    || die "download failed: $URL"
}

fetch_sha_sidecar() {
  [[ -n "$1" ]] || return 0
  curl -sSL -o "$BUNDLE_SHA" "$1" 2>/dev/null || warn "no published .sha256 next to the archive"
}

case "$SOURCE" in
  record)
    [[ -n "$RECORD" ]] || die "--record needs an id (or set ZENODO_RECORD)"
    command -v python3 >/dev/null || die "python3 is required to resolve a Zenodo record"
    info "resolving Zenodo record $RECORD"
    URL="$(zenodo_link "$RECORD" ".zip")"
    [[ -n "$URL" ]] || die "no .zip in Zenodo record $RECORD"
    fetch_sha_sidecar "$(zenodo_link "$RECORD" ".sha256")"
    fetch_archive
    ;;
  remote)
    command -v rclone >/dev/null || die "rclone is not installed: https://rclone.org/install/"
    [[ -n "$REMOTE" ]] || die "--remote needs a remote:path (or set RCLONE_REMOTE)"
    info "fetching from $REMOTE with rclone"
    rclone copy "$REMOTE/$(basename "$BUNDLE_ZIP")" "$ROOT/" --progress
    rclone copy "$REMOTE/$(basename "$BUNDLE_SHA")" "$ROOT/" 2>/dev/null || true
    ;;
  file)
    [[ -f "$FROM_FILE" ]] || die "no such file: $FROM_FILE"
    info "using local archive $FROM_FILE"
    cp -f "$FROM_FILE" "$BUNDLE_ZIP"
    [[ -f "$FROM_FILE.sha256" ]] && cp -f "$FROM_FILE.sha256" "$BUNDLE_SHA" || true
    ;;
  url)
    [[ -n "$URL" ]] || die "--url needs a link (or set DATA_URL)"
    fetch_archive
    [[ -n "$EXPECTED_SHA" ]] || fetch_sha_sidecar "$URL.sha256"
    ;;
esac

[[ -f "$BUNDLE_ZIP" ]] || die "$BUNDLE_ZIP was not downloaded"
info "archive: $(human "$(stat -c '%s' "$BUNDLE_ZIP")")"

# --- verify ----------------------------------------------------------------
if [[ -n "$EXPECTED_SHA" ]]; then
  have=$(sha256sum "$BUNDLE_ZIP" | cut -d' ' -f1)
  [[ "$have" == "$EXPECTED_SHA" ]] || die "sha256 mismatch: expected $EXPECTED_SHA, got $have"
  info "sha256 matches ($have)"
elif [[ -f "$BUNDLE_SHA" ]]; then
  # Compare hashes instead of 'sha256sum -c': the published file names the
  # remote basename, which may differ from the local one (--out).
  expected=$(awk 'NR==1 {print $1}' "$BUNDLE_SHA")
  have=$(sha256sum "$BUNDLE_ZIP" | awk '{print $1}')
  [[ ${#expected} -eq 64 ]] || die "unreadable sha256 file: $BUNDLE_SHA"
  [[ "$expected" == "$have" ]] || die "sha256 mismatch: expected $expected, got $have"
  info "sha256 matches ($have)"
else
  warn "no sha256 available, skipping integrity check"
fi

if [[ $VERIFY_ONLY -eq 1 ]]; then
  info "--verify-only given, not unpacking"
  exit 0
fi

# --- unpack ----------------------------------------------------------------
command -v unzip >/dev/null || die "unzip is required (apt install unzip)"
[[ $FORCE -eq 1 ]] && rm -rf "$target_dir"
info "unpacking into $DEST"
unzip -q -o "$BUNDLE_ZIP" -d "$DEST"

if [[ $KEEP_ZIP -eq 0 ]]; then
  rm -f "$BUNDLE_ZIP"
  info "removed $BUNDLE_ZIP (use --keep-zip to keep it)"
fi

# --- summary ---------------------------------------------------------------
if [[ -d "$target_dir" ]]; then
  n=$(find "$target_dir" -type f | wc -l)
  bytes=$(find "$target_dir" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
  info "$target_dir: $n files, $(human "$bytes")"
  echo
  echo "top-level content:"
  du -sh "$target_dir"/* 2>/dev/null | sort -rh | head -10 || true
else
  warn "the archive has no '$DATA_DIR' entry; it unpacked into $DEST instead:"
  du -sh "$DEST"/* 2>/dev/null | sort -rh | head -10 || true
  warn "pass --data-dir <name> if the top-level directory is named differently"
fi
echo
warn "the archive intentionally excludes *.h5 demos and nested *.zip files."
echo "    demos (needed to (re)fit conditions) must be fetched separately, e.g.:"
echo "      data/scenes/ogbench/scene*/experts/*/demos.h5"
