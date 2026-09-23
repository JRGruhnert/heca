import argparse
import time
from collections.abc import Iterator
from statistics import mean

import numpy as np
import torch
import matplotlib

from heca.graphs.graph import SubgoalMode
from heca.learning import dist as pdist
from scripts.common.helper import fmt_duration, generate_clients

matplotlib.use("Agg")

from heca.agents.heca import Heca
from heca.misc import logger
from scripts.common.args import add_heca_arguments, generate_tag, generate_group
from scripts.common.scenes import scene_tags

import conf.networks

SUCCESS = "stats/success_rate"
EMA_DECAY = 0.99

DEFAULTS: dict[str, object] = {
    "tag": "final",
    "scene": None,  # None = every scene is a client (omit --scene for real FL)
    "smode": SubgoalMode.BOTH,
    "wandb": True,
    "batch": 1000,
    "repeats": 3,
    "gt": True,
    "virtual": True,
    "federated": True,
    "method": "fedprox",
}

RUNS: list[dict[str, object]] = [
    {"network": "a0"},
    # {"network": "a1"},
    # {"network": "a2"},
    # {"network": "a3"},
    # {"network": "a4"},
    # {"network": "a0", "method": "fedavg"},
    # {"network": "a0", "method": "ditto", "personal_coef": 0.1, "mu": 0.0},
]


def _argparse_defaults() -> dict[str, object]:
    parser = argparse.ArgumentParser(add_help=False)
    add_heca_arguments(parser)
    return {action.dest: action.default for action in parser._actions}


def get_args() -> Iterator[argparse.Namespace]:
    """Yield one seeded Namespace per run (and per repeat) of the table."""
    base = _argparse_defaults()
    for index, row in enumerate(RUNS):
        fields = {**base, **DEFAULTS, **row}
        repeats = int(fields.pop("repeats", 1))  # type: ignore

        unknown = set(fields) - set(base)
        if unknown:
            raise KeyError(
                f"RUNS[{index}] has unknown key(s) {sorted(unknown)}; "
                f"valid keys are {sorted(base)} plus 'repeats'"
            )
        for i in range(repeats):
            run_fields = dict(fields)
            run_fields["seed"] = i
            yield argparse.Namespace(**run_fields)


class SuccessTracker:
    """Success-rate EMA (decay ``EMA_DECAY``) plus the peak of the run."""

    def __init__(self, decay: float = EMA_DECAY):
        self.decay = decay
        self.ema: float | None = None
        self.best: float | None = None
        self.updates = 0

    def update(self, metrics: dict[str, float]) -> None:
        value = metrics.get(SUCCESS)
        if value is None:
            return
        self.ema = (
            value
            if self.ema is None
            else self.decay * self.ema + (1.0 - self.decay) * value
        )
        self.best = value if self.best is None else max(self.best, value)
        self.updates += 1


def selected_scenes(args) -> list[str]:
    """The scene tags (one client each) this run trains on; imports nothing."""
    return [tag for tag in scene_tags() if not args.scene or tag == args.scene]


def train(agent: Heca, tracker: SuccessTracker, n_batch: int) -> int:
    """Train one client for ``n_batch`` episodes; each syncs with its peers."""
    n = 0
    while n < n_batch:
        if agent.tick():
            n += 1
            agent.learner.sync()
            tracker.update(dict(agent.learner.metrics))
    return n


def worker(rank: int, world_size: int, args, tag: str, group: str) -> None:
    """One client, one process: the same run as before, minus the shared server."""
    pdist.pin_intra_op_threads()
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    hecas = generate_clients(
        tag,
        group,
        conf.networks.get(args.network),
        selected_scenes(args),
        federated=True,
        method=args.method,
        personal_coef=args.personal_coef,
        mu=args.mu,
        inference=args.inference,
        virtual=args.virtual,
        use_wandb=args.wandb,
        reload=args.reload,
        use_gt=args.gt,
        smode=args.smode,
        n_batch=args.batch,
        rank=rank,
        world_size=world_size,
    )
    agent = Heca.get(hecas[0])
    tracker = SuccessTracker()
    failed = True
    try:
        n = train(agent, tracker, args.batch)
        failed = False
    finally:
        run_ = agent.learner.run
        if run_ is not None and tracker.ema is not None and tracker.best is not None:
            run_.summary[f"success/ema{EMA_DECAY}"] = tracker.ema
            run_.summary["success/max"] = tracker.best
        # exit_code=1 marks a crashed run instead of a finished one
        agent.learner.finish(exit_code=1 if failed else 0)

    scene = agent.learner.cfg.tag
    if tracker.ema is None or tracker.best is None:
        logger.warning(f"[{tag}] {scene} logged no {SUCCESS}")
    else:
        logger.info(
            f"[{tag}] rank {rank}/{world_size} {scene}: {n} update(s), "
            f"success ema({EMA_DECAY}) = {tracker.ema:.4f} | max = {tracker.best:.4f}"
        )
    ema = torch.tensor([tracker.ema if tracker.ema is not None else float("nan")])
    ema = pdist.mean_tensors({"ema": ema})["ema"]
    if pdist.is_main() and not ema.isnan():
        logger.info(
            f"[{tag}] {world_size} client(s): "
            f"mean success ema({EMA_DECAY}) = {ema.item():.4f}"
        )


def main():

    planned = list(get_args())

    logger.info(f"{len(planned)} run(s) planned:")
    for i, args in enumerate(planned, 1):
        logger.info(f"  [{i:>3}/{len(planned)}] {generate_tag(args)}")

    durations: list[float] = []
    sweep_start = time.perf_counter()

    for i, args in enumerate(planned, 1):
        logger.info(f"[{i}/{len(planned)}] starting {generate_tag(args)}")
        started = time.perf_counter()

        scenes = selected_scenes(args)
        n_ranks = args.ranks or len(scenes)
        if n_ranks != len(scenes):
            raise SystemExit(
                f"--ranks {n_ranks} but {len(scenes)} clients; one process per "
                "client is required. Pass --scene to train a single client."
            )
        pdist.spawn(
            worker,
            n_ranks,
            args=(args, generate_tag(args), generate_group(args)),
        )

        durations.append(time.perf_counter() - started)
        elapsed = time.perf_counter() - sweep_start
        remaining = len(planned) - i
        report = (
            f"  [{i}/{len(planned)}] {generate_tag(args)} took "
            f"{fmt_duration(durations[-1])} | {i}/{len(planned)} done in "
            f"{fmt_duration(elapsed)}"
        )
        if remaining:
            avg = mean(durations)
            report += (
                f" | eta {fmt_duration(avg * remaining)} for the remaining "
                f"{remaining} (avg {fmt_duration(avg)}/run)"
            )
        else:
            report += " | sweep complete"
        logger.info(report)


if __name__ == "__main__":
    main()

# good scenes: 0, 1, 8
# bad scenes: 2, 4
# fixable: 3, 5, 6, 7, 9, 10
