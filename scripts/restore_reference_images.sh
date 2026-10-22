#!/usr/bin/env bash
# Put a scene's reference images (pos.png) back from the data bundle archive.
#
#   ./scripts/restore_reference_images.sh [scene] [archive] [--all]
#   ./scripts/restore_reference_images.sh scene0 ~/heca/data3.tar.zst
#
# Only the images are touched. annotation.json keeps the clicks (x, y, xyz, frame)
# and the slide ends an expert pipeline wrote into it, so it is never replaced.
# A missing or unreadable image is restored; an intact one is left alone unless
# --all is passed, because a reference re-clicked after the archive was made would
# otherwise be paired with a stale image.
#
# HECA_REFS_DEST  where the scene lives  (default: data/scenes/ogbench/<scene>)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

scene="scene0"
archive="$HOME/heca/data3.tar.zst"
force=""
while (( $# )); do
    case "$1" in
        --all) force=1; shift ;;
        -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) if [[ "$1" == *.tar.zst || "$1" == *.tar* || -f "$1" ]]; then
               archive="$1"
           else
               scene="$1"
           fi
           shift ;;
    esac
done

[[ -f "$archive" ]] || { echo "error: no archive at $archive" >&2; exit 1; }
dest="${HECA_REFS_DEST:-data/scenes/ogbench/$scene}"
[[ -d "$dest" ]] || { echo "error: no scene directory $dest" >&2; exit 1; }
command -v zstd >/dev/null || { echo "error: zstd is not on PATH" >&2; exit 1; }

# the image check needs PIL, which lives in the project env, not in the login shell
py="${PYTHON:-python}"
if ! command -v "$py" >/dev/null || ! "$py" -c 'import PIL' 2>/dev/null; then
    for candidate in "$HOME/miniconda3/envs/hecarim/bin/python" \
                     "$HOME/anaconda3/envs/hecarim/bin/python"; do
        if [[ -x "$candidate" ]] && "$candidate" -c 'import PIL' 2>/dev/null; then
            py="$candidate"
            break
        fi
    done
fi
"$py" -c 'import PIL' 2>/dev/null || {
    echo "error: no python with PIL found; set PYTHON=/path/to/envs/hecarim/bin/python" >&2
    exit 1
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "=== $archive -> $dest"
tar -I zstd -C "$tmp" -xf "$archive" --wildcards "*scenes/ogbench/$scene/*/pos.png"
# -C must precede the pattern: it only affects file names that follow it, and with
# the pattern first tar silently extracts nothing (exit 0, no files)
mapfile -t found < <(find "$tmp" -name pos.png | sort)
if (( ${#found[@]} == 0 )); then
    echo "error: the archive holds no *scenes/ogbench/$scene/*/pos.png" >&2
    echo "       check with: tar -I zstd -tf $archive | grep 'pos\\.png'" >&2
    exit 1
fi
echo "=== ${#found[@]} image(s) in the archive for $scene"

"$py" - "$dest" "$force" "${found[@]}" <<'PY'
import shutil
import sys
from pathlib import Path

from PIL import Image

dest = Path(sys.argv[1])
force = bool(sys.argv[2])
sources = {Path(p).parent.name: Path(p) for p in sys.argv[3:]}


def load(path: Path):
    try:
        image = Image.open(path)
        image.load()
        return image
    except Exception:
        return None


restored, skipped, broken, orphan = [], [], [], []
for label, source in sorted(sources.items()):
    target = dest / label / "pos.png"
    if load(source) is None:
        broken.append(label)
        continue
    if not (target.parent / "annotation.json").exists():
        # an image without its click cannot be used: the x/y/xyz/frame live there
        orphan.append(label)
        continue
    if target.exists() and not force and load(target) is not None:
        skipped.append(label)
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    (restored if load(target) is not None else broken).append(label)

print(f"restored:    {', '.join(restored) or '-'}")
print(f"already fine:{' ' + ', '.join(skipped) if skipped else ' -'}")
if orphan:
    print(f"no annotation: {', '.join(orphan)} (image skipped, the click is gone)")
if broken:
    print(f"unreadable:  {', '.join(broken)}")

print("\n=== every reference image in the scene now:")
for image in sorted(dest.glob("*/pos.png")):
    loaded = load(image)
    state = f"{loaded.size[0]}x{loaded.size[1]} {loaded.mode}" if loaded else "BROKEN"
    note = (image.parent / "annotation.json").exists() and "click" or "NO CLICK"
    print(f"  {image.parent.name:9s} {image.stat().st_size:7d} B  {state:12s} {note}")

raise SystemExit(1 if broken or orphan else 0)
PY
