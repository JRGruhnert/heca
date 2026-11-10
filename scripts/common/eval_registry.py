"""Per-network/mode eval registries: every evaluated run in one file.

One registry per (network, fed, eval mode, container):

    <root>/eval/{network}[_fed]_{mode}[_container].json

Each run is one entry under ``runs``, keyed by its run-dir name, carrying the
checkpoint it was evaluated at and its result. The ``protocols`` header is
recomputed on every save: one across-seed aggregate per observation protocol
(gt / gt-virt / vis), so a file may hold several protocols without mixing their
statistics.
"""
import json
from pathlib import Path
from statistics import mean, stdev

from typing import Any


def registry_path(
    root: Path, network: str, fed: bool, mode: str, container: bool
) -> Path:
    name = (
        f"{network}"
        f"{'_fed' if fed else ''}"
        f"_{mode}"
        f"{'_container' if container else ''}.json"
    )
    return root / "eval" / name


def load_registry(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            # only a registry (with a ``runs`` map) is loaded; an old-style
            # summary file at the same path starts fresh instead of mixing
            if isinstance(data, dict) and isinstance(data.get("runs"), dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"runs": {}}


def _task_aggregate(pairs: list[tuple[str, dict]]) -> list[dict]:
    """Per-task success rates across seeds, one entry per task.

    ``pairs`` is the (run name, run entry) list of one protocol; only entries
    that carry a ``tasks`` list (task-mode evaluations) contribute.
    """
    cells: dict[tuple, dict] = {}
    for name, entry in pairs:
        for task in entry.get("tasks") or []:
            key = (task.get("task_id"), str(task.get("task_name", "")))
            cell = cells.setdefault(
                key,
                {
                    "task_id": task.get("task_id"),
                    "task_name": str(task.get("task_name", "")),
                    "success_rate_per_seed": {},
                },
            )
            cell["success_rate_per_seed"][name] = float(task["success_rate"])
            per_epoch = cell.setdefault("per_epoch", {})
            for epoch, rate in enumerate(task.get("success_rate_per_epoch") or []):
                per_epoch.setdefault(epoch, []).append(float(rate))
    out = []
    for (task_id, _), cell in sorted(
        cells.items(), key=lambda item: (item[0][0] is None, item[0][0] or 0)
    ):
        rates = list(cell["success_rate_per_seed"].values())
        per_epoch = cell.pop("per_epoch", {})
        out.append(
            {
                **cell,
                "mean": mean(rates),
                "std": stdev(rates) if len(rates) > 1 else 0.0,
                "n_seeds": len(rates),
                **(
                    {
                        "success_rate_per_epoch": [
                            mean(per_epoch[e]) for e in range(max(per_epoch) + 1)
                        ]
                    }
                    if per_epoch
                    else {}
                ),
            }
        )
    return out


def save_registry(path: Path, registry: dict[str, Any]) -> None:
    """Write the registry with its aggregate header recomputed from ``runs``."""
    protocols: dict[str, list[tuple[str, dict]]] = {}
    for name, entry in registry.get("runs", {}).items():
        key = ("gt" if entry.get("gt") else "") + (
            "-virt" if entry.get("virtual") else ""
        )
        key = key or "vis"
        protocols.setdefault(key, []).append((name, entry))

    aggregates: dict[str, dict] = {}
    for key, pairs in sorted(protocols.items()):
        rates = [float(entry["success_rate"]) for _, entry in pairs]
        if not rates:
            continue
        aggregate = {
            "n_seeds": len(pairs),
            "mean": mean(rates),
            "std": stdev(rates) if len(rates) > 1 else 0.0,
            "success_rate_per_seed": {
                name: float(entry["success_rate"]) for name, entry in pairs
            },
        }
        if any(entry.get("tasks") for _, entry in pairs):
            aggregate["tasks"] = _task_aggregate(pairs)
        aggregates[key] = aggregate
    registry["protocols"] = aggregates
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2) + "\n")


def is_fresh(registry: dict[str, Any], run_name: str, checkpoint: str) -> bool:
    """Whether the run already has a result at exactly this checkpoint."""
    entry = registry.get("runs", {}).get(run_name)
    return bool(entry and entry.get("checkpoint") == checkpoint)
