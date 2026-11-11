"""Sweep every network x every checkpoint: virt-container, virt-real, gt-real.

For each run (training seed) under ``--root`` this evaluates every checkpoint up
to ``--last`` (0 = latest available) under two virtual protocols, and for the
``--gt-nets`` (default a0 a10 a12) the last checkpoint under the real-stepping
protocol. Results are written to ``--out`` (default ``eval/`` at the repo root,
separate from the run folders so nothing conflicts):

    eval/raw/<tag>__<protocol>.jsonl   one line per checkpoint (resumable)
    eval/<group>.csv                   one per experiment tag (network + protocol
                                       + federation settings, seed suffix stripped):
                                       epoch, virt-container (+std), virt-real
                                       (+std), gt-real (+std), averaged over seeds
    eval/runs/<tag>.csv                per run, raw values (no averaging)
    eval/sweep.json                    the raw records, keyed by tag/protocol

Federated runs (a run folder with a ``global/`` subfolder of checkpoints) are
evaluated with the server (global) weights only; ``clients/`` are ignored.

Usage:
    python -m scripts.c07_sweep --root data/network/standard --out eval \
        --last 750 --workers 10

    # one protocol only, or a subset of runs
    python -m scripts.c07_sweep --protocols virt --last 750
    python -m scripts.c07_sweep --only a12 a10
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev

import conf.networks
from heca.misc import logger
from scripts.common.helper import fmt_duration

PROTOCOLS = ("virt-container", "virt-real", "gt-real")


def parse_run_name(name: str) -> dict:
    """Extract scene / network / seed from a run folder name.

    Works for ``scene0_final-a0-gt-virt_s2`` and federated names that carry the
    same tokens; the network token is matched exactly against the registry so
    ``a1`` and ``a10`` cannot be confused.
    """
    stem, _, seed = name.rpartition("_s")
    if not (stem and seed.isdigit()):
        stem, seed = name, ""
    scene, _, rest = stem.partition("_")
    tokens = [t for t in rest.split("-") if t]
    network = next((t for t in tokens if t in conf.networks.CONFIGS), None)
    return {
        "scene": scene or "scene0",
        "network": network,
        "seed": int(seed) if seed else None,
    }


def checkpoint_source(path: Path) -> Path | None:
    """The directory holding this run's checkpoints.

    - standard runs: the run folder itself
    - federated runs: ``<run>/clients/global`` (the server aggregate; the
      per-scene ``clients/<scene>`` copies are NOT the weights to evaluate)
    - ``<run>/global`` for runs whose aggregate was copied out separately
    """
    for candidate in (path, path / "clients" / "global", path / "global"):
        if candidate.is_dir() and any(candidate.glob("ckp_*.pt")):
            return candidate
    return None


def discover_runs(root: Path, only: list[str]) -> list[dict]:
    """Run folders under ``root``: standard or federated (server checkpoints)."""
    runs = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if only and not any(token in path.name for token in only):
            continue
        ckp_dir = checkpoint_source(path)
        if ckp_dir is None:
            continue
        parts = parse_run_name(path.name)
        if not parts["network"]:
            logger.warning(f"skipping {path.name}: no network token in the name")
            continue
        stem, _, seed = path.name.rpartition("_s")
        group = stem if (stem and seed.isdigit()) else path.name
        runs.append({"tag": path.name, "group": group, "run_dir": path,
                     "ckp_dir": ckp_dir, "network": parts["network"],
                     "seed": parts["seed"]})
    if not runs:
        raise SystemExit(f"no checkpointed runs under {root}")
    return runs


def max_epoch(run: dict) -> int:
    return max(int(p.stem.split("_")[1]) for p in run["ckp_dir"].glob("ckp_*.pt"))


def load_records(out: Path) -> dict[tuple[str, str], dict[int, dict]]:
    records: dict[tuple[str, str], dict[int, dict]] = {}
    raw = out / "raw"
    if not raw.is_dir():
        return records
    for path in raw.glob("*.jsonl"):
        tag, _, protocol = path.stem.rpartition("__")
        per_epoch: dict[int, dict] = {}
        for line in path.read_text().splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            per_epoch[int(payload["epoch"])] = payload
        records[(tag, protocol)] = per_epoch
    return records


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return mean(values), (stdev(values) if len(values) > 1 else 0.0)


def fmt(value: float) -> str:
    return "" if value != value else f"{value:.4f}"  # nan -> empty cell


def write_csvs(runs: list[dict], records: dict, out: Path, last: int, gt_nets: set[str]) -> None:
    # one CSV per experiment group (network + protocol + federation settings),
    # never mixing runs of different tags: only the seed suffix is averaged
    by_group: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        by_group.setdefault((run["network"], run["group"]), []).append(run)

    for (network, group), net_runs in sorted(by_group.items()):
        virt: dict[int, list[float]] = {}
        virt2: dict[int, list[float]] = {}
        gt: dict[int, list[float]] = {}
        epochs: set[int] = set()
        for run in net_runs:
            for protocol, bucket in (
                ("virt-container", virt),
                ("virt-real", virt2),
                ("gt-real", gt),
            ):
                recs = records.get((run["tag"], protocol), {})
                for epoch, payload in recs.items():
                    epochs.add(epoch)
                    bucket.setdefault(epoch, []).append(payload["success_rate"])

        pending_gt = bool(network in gt_nets) and last and last not in gt

        rows = []
        for epoch in sorted(epochs | set(gt)):
            vc_mean, vc_std = mean_std(virt.get(epoch, []))
            vr_mean, vr_std = mean_std(virt2.get(epoch, []))
            gt_mean, gt_std = mean_std(gt.get(epoch, []))
            rows.append((epoch, vc_mean, vc_std, vr_mean, vr_std, gt_mean, gt_std))

        path = out / f"{group}.csv"
        lines = ["epoch,virt-container,virt-container-std,virt-real,virt-real-std,gt-real,gt-real-std"]
        for row in rows:
            lines.append(",".join([str(row[0]), fmt(row[1]), fmt(row[2]), fmt(row[3]), fmt(row[4]), fmt(row[5]), fmt(row[6])]))
        path.write_text("\n".join(lines) + "\n")
        logger.info(
            f"{network}: {len(rows)} epoch row(s) from {len(net_runs)} run(s) -> {path}"
            + (" (gt-real pending)" if pending_gt else "")
        )

    # one file per run (seed / federated variant), no averaging
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for run in sorted(runs, key=lambda r: r["tag"]):
        per_epochs: dict[int, dict] = {}
        for protocol in ("virt-container", "virt-real", "gt-real"):
            for epoch, payload in records.get((run["tag"], protocol), {}).items():
                per_epochs.setdefault(epoch, {})[protocol] = payload["success_rate"]
        if not per_epochs:
            continue
        lines = ["epoch,virt-container,virt-real,gt-real"]
        for epoch in sorted(per_epochs):
            cell = per_epochs[epoch]
            lines.append(
                ",".join(
                    [str(epoch)]
                    + [fmt(cell.get(p, float("nan"))) for p in ("virt-container", "virt-real", "gt-real")]
                )
            )
        (runs_dir / f"{run['tag']}.csv").write_text("\n".join(lines) + "\n")

    (out / "sweep.json").write_text(
        json.dumps(
            {
                "last": last,
                "generated": str(__import__("datetime").datetime.now().isoformat(timespec="seconds")),
                "runs": [
                    {"tag": r["tag"], "network": r["network"], "seed": r["seed"]}
                    for r in runs
                ],
                "records": {
                    f"{tag}__{protocol}": per_epoch
                    for (tag, protocol), per_epoch in records.items()
                },
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/network/standard"))
    ap.add_argument("--out", type=Path, default=Path("eval"))
    ap.add_argument("--last", type=int, default=0, help="max checkpoint epoch, 0 = latest")
    ap.add_argument("--per-task", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only", nargs="+", default=[], help="keep runs whose name contains one of these")
    ap.add_argument(
        "--gt-nets",
        nargs="+",
        default=["all"],
        help="networks for the real-stepping protocol, or 'all' for every network",
    )
    ap.add_argument(
        "--gt-epochs",
        nargs="+",
        default=["last"],
        help="checkpoints for the real-stepping protocol: 'last' or explicit epochs",
    )
    ap.add_argument("--protocols", choices=("all", "virt", "gt"), default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--csv-only",
        action="store_true",
        help="skip the runs and only aggregate the raw files already in --out/raw",
    )
    args = ap.parse_args()

    runs = discover_runs(args.root, args.only)
    last = args.last or max(max_epoch(run) for run in runs)
    args.out.mkdir(parents=True, exist_ok=True)
    networks = sorted({run["network"] for run in runs})
    gt_nets = set(networks) if "all" in set(args.gt_nets) else set(args.gt_nets)
    gt_epochs = [last if token == "last" else int(token) for token in args.gt_epochs]
    logger.info(
        f"{len(runs)} run(s), {len(networks)} network(s) {networks}, "
        f"checkpoints <= {last}, protocols={args.protocols}, "
        f"gt nents={sorted(gt_nets)} gt epochs={gt_epochs}"
    )

    jobs: list[tuple[str, dict, int | None]] = []
    for run in runs:
        if args.protocols in ("all", "virt"):
            for protocol in ("virt-container", "virt-real"):
                jobs.append((protocol, run, None))
    # the long gt jobs go last so the pool does not starve the virt sweep
    if args.protocols in ("all", "gt"):
        for run in runs:
            if run["network"] in gt_nets:
                for epoch in gt_epochs:
                    jobs.append(("gt-real", run, epoch))

    if args.dry_run:
        for protocol, run, epoch in jobs:
            extra = f" --epoch {epoch}" if epoch else ""
            logger.info(f"  {run['tag']} [{protocol}]{extra}")
        return

    if args.csv_only:
        records = load_records(args.out)
        write_csvs(runs, records, args.out, last, gt_nets)
        logger.info(f"csv-only: {len(records)} raw file(s) aggregated into {args.out}")
        return

    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")

    def launch(job: tuple[str, dict, int | None]) -> tuple[str, dict, bool]:
        protocol, run, epoch = job
        cmd = [
            sys.executable, "-m", "scripts.c07_sweep_worker",
            "--run-dir", str(run["run_dir"]),
            "--ckp-dir", str(run["ckp_dir"]),
            "--tag", run["tag"],
            "--network", run["network"],
            "--protocol", protocol,
            "--last", str(last),
            "--per-task", str(args.per_task),
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
            "--out", str(args.out),
        ]
        if epoch:
            cmd += ["--epoch", str(epoch)]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"{run['tag']} [{protocol}] failed:\n{result.stderr[-3000:]}")
        return protocol, run, result.returncode == 0

    import time
    started = time.perf_counter()
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(launch, job): job for job in jobs}
        for future in as_completed(futures):
            protocol, run, ok = future.result()
            done += 1
            failed += int(not ok)
            if done % 10 == 0 or done == len(jobs):
                logger.info(
                    f"jobs {done}/{len(jobs)} (failed {failed}) "
                    f"in {fmt_duration(time.perf_counter() - started)}"
                )

    records = load_records(args.out)
    write_csvs(runs, records, args.out, last, gt_nets)
    logger.info(
        f"done: {done} job(s), {failed} failed, "
        f"{len(records)} raw file(s) in {args.out}/raw"
    )


if __name__ == "__main__":
    main()
