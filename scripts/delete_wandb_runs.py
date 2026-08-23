"""Delete wandb runs from the cloud AND from the local disk together.

Deleting a run in the web UI (or via ``run.delete()``) only removes the cloud
copy; the local ``wandb/run-*`` folders on disk are untouched. This script
deletes both: for every run matching the filters it removes the run from the
wandb server and deletes the matching local run folder(s) (which embed the
run id, e.g. ``wandb/run-20260823_213051-a1b2c3d4``).

Always starts in *dry-run* mode: it only prints what would be deleted. Pass
``--delete`` to actually delete.

Run from the repo root (in an env with wandb installed and logged in)::

    python scripts/delete_wandb_runs.py --group exp
    python scripts/delete_wandb_runs.py --group exp --name-prefix exp_scene1
    python scripts/delete_wandb_runs.py --group exp --delete
    python scripts/delete_wandb_runs.py --group exp --wandb-dir /path/to/wandb --delete
"""

import argparse
import shutil
import sys
from pathlib import Path

# Make ``conf`` / ``scripts.common`` importable when run directly as
# ``python scripts/delete_wandb_runs.py`` (mirrors scripts/__init__.py).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from heca.learning.learner import WandBConfig

try:
    import wandb
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"wandb is not installed in the active environment: {exc}"
    ) from exc


def default_wandb_dir() -> Path:
    """First existing local wandb dir, else ``./wandb``."""
    for candidate in (Path.cwd() / "wandb", Path.home() / "wandb"):
        if candidate.exists():
            return candidate
    return Path.cwd() / "wandb"


def find_local_run_dirs(wandb_dir: Path, run_id: str) -> list[Path]:
    """Local run folders belonging to a cloud run id.

    Online folders are named ``run-<timestamp>-<run_id>``, offline folders
    ``offline-run-<timestamp>-<run_id>`` — the id is the final dash-separated
    segment. Also matches the ``run-<id>.wandb`` marker files' parents.
    """
    if not wandb_dir.exists():
        return []
    matches = list(wandb_dir.glob(f"run-*-{run_id}"))
    matches += list(wandb_dir.glob(f"offline-run-*-{run_id}"))
    # Fallback: any folder containing a run-<id>.wandb marker.
    if not matches:
        for marker in wandb_dir.glob(f"*/run-{run_id}.wandb"):
            matches.append(marker.parent)
    return sorted(dict.fromkeys(matches))  # dedupe, keep order


def delete_run(run, wandb_dir: Path, dry_run: bool) -> dict:
    """Delete one run from the cloud and its local folders. Returns a report."""
    local_dirs = find_local_run_dirs(wandb_dir, run.id)
    report = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "cloud_deleted": False,
        "local_deleted": [str(d) for d in local_dirs],
    }
    prefix = "DRY-RUN (would delete)" if dry_run else "DELETING"

    if not dry_run:
        run.delete()
        report["cloud_deleted"] = True
        for d in local_dirs:
            shutil.rmtree(d, ignore_errors=True)

    print(f"  {prefix}: {run.name} (id={run.id}, state={run.state})")
    for d in local_dirs:
        print(f"      local folder: {d}")
    if not local_dirs:
        print(f"      (no local folder found for id {run.id} under {wandb_dir})")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity",
        default=WandBConfig().entity,
        help=f"wandb entity (default: {WandBConfig().entity}).",
    )
    parser.add_argument(
        "--project",
        default=WandBConfig().project,
        help=f"wandb project (default: {WandBConfig().project}).",
    )
    parser.add_argument(
        "--group",
        default=None,
        help="Only delete runs with this wandb group (e.g. exp).",
    )
    parser.add_argument(
        "--name-prefix",
        default=None,
        help="Only delete runs whose name starts with this prefix.",
    )
    parser.add_argument(
        "--wandb-dir",
        default=None,
        help="Local wandb directory (default: auto-detect ./wandb or ~/wandb).",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete. Without this flag the script only prints "
        "what it would delete (dry-run).",
    )
    args = parser.parse_args()

    filters = {}
    if args.group:
        filters["group"] = args.group
    runs = wandb.Api().runs(
        f"{args.entity}/{args.project}", filters=filters or None
    )
    if args.name_prefix:
        runs = [r for r in runs if r.name.startswith(args.name_prefix)]
    if not runs:
        raise SystemExit(
            f"No runs found for entity={args.entity} project={args.project} "
            f"group={args.group!r} name_prefix={args.name_prefix!r}."
        )

    wandb_dir = Path(args.wandb_dir) if args.wandb_dir else default_wandb_dir()
    print(
        f"{'DRY-RUN' if not args.delete else 'DELETE'} mode — "
        f"{len(runs)} run(s) in {args.entity}/{args.project}, "
        f"local dir: {wandb_dir}"
    )
    if not wandb_dir.exists():
        print(f"  (local wandb dir {wandb_dir} does not exist)")

    reports = [delete_run(run, wandb_dir, dry_run=not args.delete) for run in runs]

    if not args.delete:
        print(
            "\nDry-run finished — nothing was deleted. Re-run with --delete "
            "to actually delete."
        )
    else:
        n_cloud = sum(r["cloud_deleted"] for r in reports)
        n_local = sum(len(r["local_deleted"]) for r in reports)
        print(f"\nDeleted {n_cloud} run(s) from the cloud and "
              f"{n_local} local folder(s).")


if __name__ == "__main__":
    main()
