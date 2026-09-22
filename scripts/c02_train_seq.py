import argparse
import time
from collections.abc import Iterator
from statistics import mean

import numpy as np
import torch
import matplotlib

from heca.graphs.graph import SubgoalMode
from heca.heca_gnn.network import Network
from scripts.common.helper import fmt_duration, generate_clients

matplotlib.use("Agg")

from heca.agents.heca import Heca
from heca.experts.expert import ExpertModel
from heca.learning.ppo import PPO
from heca.misc import logger
from scripts.common.args import add_heca_arguments, generate_tag, generate_group
from scripts.common.scenes import find_scene_models

import conf.networks

SUCCESS = "stats/success_rate"
EMA_DECAY = 0.99

DEFAULTS: dict[str, object] = {
    "tag": "final",
    "scene": "scene0",
    "smode": SubgoalMode.BOTH,
    "wandb": True,
    "batch": 1000,
    "repeats": 3,
    "gt": True,
    "virtual": True,
}

RUNS: list[dict[str, object]] = [
    {"network": "a0"},
    {"network": "a1"},
    {"network": "a2"},
    {"network": "a3"},
    {"network": "a4"},
    # {"network": "x5"},
    # {"network": "x6"},
    # {"network": "x7"},
    # {"network": "x8"},
    # {"network": "x9"},
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


def run(agent: Heca, n_batch: int = 1000, on_update=None):
    """Train ``n_batch`` updates; ``on_update`` sees the metrics of each one."""
    n = 0
    while n < n_batch:
        if agent.tick():
            n += 1
            agent.learner.sync()
            if on_update is not None:
                on_update(dict(agent.learner.metrics))
    return n


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

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        exps, _ = generate_clients(
            generate_tag(args),
            generate_group(args),
            conf.networks.get(args.network),
            federated=False,
            clients={str(args.scene): find_scene_models(args.scene)},
            inference=args.inference,
            virtual=args.virtual,
            use_wandb=args.wandb,
            reload=args.reload,
            use_gt=args.gt,
            smode=args.smode,
            n_batch=args.batch,
        )
        exp = exps[0]
        tracker = SuccessTracker()
        agent = Heca.get(exp)
        failed = True
        try:
            run(agent, args.batch, on_update=tracker.update)

            if tracker.ema is None or tracker.best is None:
                logger.warning(
                    f"  [{i}/{len(planned)}] {generate_tag(args)} logged no {SUCCESS}"
                )
            else:
                logger.info(
                    f"  [{i}/{len(planned)}] {generate_tag(args)}: "
                    f"success ema({EMA_DECAY}) = {tracker.ema:.4f} | "
                    f"max = {tracker.best:.4f} over {tracker.updates} update(s)"
                )
                if agent.learner.run is not None:
                    agent.learner.run.summary[f"success/ema{EMA_DECAY}"] = tracker.ema
                    agent.learner.run.summary["success/max"] = tracker.best
            failed = False
        finally:
            if agent:
                # exit_code=1 marks a crashed run instead of a finished one
                agent.learner.finish(exit_code=1 if failed else 0)

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
