import sys
from pathlib import Path

# ``conf`` is a generic top-level package name and can be shadowed by an
# unrelated ``conf`` package installed in site-packages. Make sure the repo's
# ``conf`` package wins when these scripts are run as ``python scripts/...``.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)
