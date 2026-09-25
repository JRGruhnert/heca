import argparse
import gc
import heapq
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")

import conf.networks

from heca.agents.heca import Heca
from heca.data.data import DCScene
from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.graphs.roles import ENMode
from heca.learning.ppo import PPO
from heca.misc import logger
from heca.misc.base import Configurable
from heca.scenes.scene import Scene

from scripts.common.args import (
    add_scene_argument,
    add_seed_argument,
    add_smode_argument,
    add_use_gt_argument,
    add_virtual_argument,
)
from scripts.common.helper import fmt_duration
from scripts.common.scenes import agents_by_scene

STATE_DIGITS = 3  # rounding used to deduplicate states
# Measured: one search node (scene copy + qpos/qvel/object snapshot) is ~2.73 KB,
# which is how --max-ram-mb turns into a frontier size.
NODE_BYTES = 2730
PROGRESS_EVERY = 25  # episodes between two progress lines


# --- search ----------------------------------------------------------------


def state_key(x: DCScene, labels: list[str]) -> tuple:
    """Rounded, hashable signature of a scene's entity values (dedup key)."""
    return tuple(
        np.round(np.asarray(x.get(label).value, dtype=float), STATE_DIGITS).tobytes()
        for label in labels
    )


@dataclass(slots=True)
class Node:
    x: DCScene
    budget: float
    depth: int
    snap: tuple


@dataclass(slots=True)
class Result:
    solved: bool
    capped: bool = False
    rollouts: int = 0  # option rollouts spent on this episode
    states: int = 0  # distinct states reached
    frontier: int = 0  # largest the frontier got, in entries


def option_reach(graph: Graph) -> tuple[dict[int, int], int]:
    """How many objects each option can complete in one application, and the max.

    The env only teleports the entities of the subgoal it is handed, i.e. the
    rows carrying a target mode; anchors keep their current value. That bound is
    what makes the heuristic below admissible.
    """
    reach: dict[int, int] = {}
    for index, option in enumerate(graph.ns_option.items):
        entities = {
            graph.ns_entity.get_by_key(key).entity
            for key in option.sources.get(EntityNodes.type, ())
            if graph.ns_entity.get_by_key(key).mode
            in (ENMode.GOAL, ENMode.SAMPLE, ENMode.SUBGOAL)
        }
        reach[index] = max(len(entities), 1)
    return reach, max(reach.values(), default=1)


def heuristic(unsatisfied: int, reach: int) -> int:
    """Least number of further options: each one completes at most ``reach``."""
    return -(-unsatisfied // reach)


def feasible_options(graph: Graph, x: DCScene, y: DCScene, budget: float) -> list[int]:
    """The option indices that pass the gate for this state."""
    data = graph.build(x, y, budget)
    return np.flatnonzero(data.option.gated.numpy() == 0).tolist()


def action_signature(graph: Graph, index: int, targets: list[str] | None) -> tuple:
    """Which entities an option teleports and which values they get.

    A virtual step is determined by exactly that, so options with an equal
    signature have equal successors and only one of them needs a rollout.
    """
    if targets is None:  # a rollout of a learned policy is an action of its own
        return (index,)
    sources = []
    for key in graph.ns_option.idx_get(index).sources[EntityNodes.type]:
        node = graph.ns_entity.get_by_key(key)
        if node.entity in targets:
            sources.append((node.entity, node.data.value.tobytes()))
    return tuple(targets), tuple(sorted(sources))


def trim(heap: list, max_frontier: int) -> int:
    """Drop the worst frontier entries so it fits the RAM cap; returns how many."""
    if not max_frontier or len(heap) <= max_frontier:
        return 0
    extra = len(heap) - max_frontier
    heap[:] = heapq.nsmallest(max_frontier, heap)
    heapq.heapify(heap)
    return extra


def search(
    scene: Scene,
    graph: Graph,
    x: DCScene,
    y: DCScene,
    labels: list[str],
    max_steps: int,
    max_nodes: int,
    max_frontier: int,
    max_reach: int,
) -> Result:
    """Can ``x`` reach an env success state within ``max_steps`` options?"""
    start = Node(x=x, budget=0.0, depth=0, snap=scene.snapshot())
    unsatisfied = sum(1 for ok in scene.successes().values() if not ok)
    n_objects = len(scene.successes())
    ties = itertools.count()
    heap = [(heuristic(unsatisfied, max_reach), 0, next(ties), start)]
    best_g: dict[tuple, int] = {state_key(x, labels): 0}
    nodes = 0
    capped = False
    frontier = 0
    teleport: dict[int, list[str] | None] = {}

    while heap:
        if nodes >= max_nodes:
            capped = True
            break
        _, popped_g, _, node = heapq.heappop(heap)
        if popped_g > best_g.get(state_key(node.x, labels), popped_g):
            continue  # superseded: the same state was reached more cheaply
        if node.depth >= max_steps:
            continue

        scene.restore(node.snap, node.depth)
        batch: dict[tuple, int] = {}
        for index in feasible_options(graph, node.x, y, node.budget):
            if index not in teleport:
                expert = ExpertModel.get(graph.ns_option.idx_get(index).model)
                teleport[index] = (
                    sorted(expert.conditions.target_entities)
                    if expert.act_virtual
                    else None
                )
            batch.setdefault(action_signature(graph, index, teleport[index]), index)

        for index in batch.values():
            scene.restore(node.snap, node.depth)
            agent_cfg, subgoal = graph.select(index)
            z, fb = ExpertModel.get(agent_cfg).act(node.x, subgoal, gated=False)
            nodes += 1

            if scene.success():
                return Result(True, capped, nodes, len(best_g), frontier)
            if fb.end:  # episode over without success: nothing to expand
                continue

            depth = node.depth + 1
            key = state_key(z, labels)
            if best_g.get(key, depth + 1) <= depth:
                continue  # this state was already reached as cheaply
            best_g[key] = depth
            heapq.heappush(
                heap,
                (
                    depth
                    + heuristic(n_objects - sum(scene.successes().values()), max_reach),
                    depth,
                    next(ties),
                    Node(x=z, budget=fb.budget, depth=depth, snap=scene.snapshot()),
                ),
            )
            frontier = max(frontier, len(heap))

        # a trimmed frontier can no longer prove that nothing exists
        capped = capped or trim(heap, max_frontier) > 0

    return Result(False, capped, nodes, len(best_g), frontier)


# --- bookkeeping -----------------------------------------------------------


def available_gb() -> float:
    """RAM the kernel is willing to hand out before it starts swapping."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / (1024 * 1024)
    return float("inf")


def clear_caches() -> None:
    """Drop the memoised scenes/experts so the next scene starts clean.

    ``Scene.get``/``ExpertModel.get`` keep every instance alive in class-level
    ``_instances`` dicts, so without this a scene loop grows by a few hundred MB
    per scene (the garbage collector cannot help: they are still referenced).
    """
    stack, seen = [Configurable], set()
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        getattr(cls, "_instances", {}).clear()
        getattr(cls, "_persisted_instances", {}).clear()
        stack.extend(cls.__subclasses__())


def write_summary(
    path: Path,
    results: list[Result],
    scene_tag: str,
    args: argparse.Namespace,
    horizon: int,
    elapsed: float,
    partial: bool,
) -> dict:
    solved = sum(r.solved for r in results)
    summary = {
        "scene": scene_tag,
        "seed": args.seed,
        "smode": str(args.smode),
        "virtual": args.virtual,
        "gt": args.gt,
        "horizon": horizon,
        "episodes": len(results),
        "episodes_requested": args.episodes,
        "solved": solved,
        "reachable": solved / len(results) if results else 0.0,
        "capped": sum(r.capped for r in results),
        "partial": partial,
        "elapsed_s": round(elapsed, 1),
        # per-episode outcome, so a resumed run keeps what it already measured
        "episode_solved": [r.solved for r in results],
        "episode_capped": [r.capped for r in results],
        "episode_rollouts": [r.rollouts for r in results],
        "episode_states": [r.states for r in results],
        "episode_frontier": [r.frontier for r in results],
        "peak_frontier": max((r.frontier for r in results), default=0),
    }
    path.write_text(json.dumps(summary, indent=2))
    return summary


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def build_heca(scene_cfg, model_cfgs, args) -> Heca:
    """The training entry point, so the graph is exactly the training graph."""
    return Heca.get(
        Heca.Config(
            experts=model_cfgs,
            learner=PPO.Config(
                tag=scene_cfg.tag,
                network=conf.networks.CONFIGS[conf.networks.NETWORK_NAMES[0]],
                wandb=logger.WandBConfig(enabled=False),
                max_update=0,
                lr_annealing=False,
            ),
            visualize=False,
            inference=True,
            virtual=args.virtual,
            reload=False,
            use_gt=args.gt,
            smode=args.smode,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)  # omitted = every scene, in order
    add_smode_argument(parser)
    add_use_gt_argument(parser)
    add_virtual_argument(parser)
    add_seed_argument(parser)
    parser.set_defaults(gt=True, virtual=True, seed=0)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument(
        "--max-nodes", type=int, default=100000, help="expansions per episode"
    )
    parser.add_argument(
        "--max-ram-mb",
        type=int,
        default=4048,
        help="RAM cap for one episode's frontier "
        f"(~{512 * 1024 * 1024 // NODE_BYTES} nodes at {NODE_BYTES / 1024:.2f} KB/node)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=6.0,
        help="wait before a scene until this much RAM is free",
    )
    args = parser.parse_args()

    max_frontier = args.max_ram_mb * 1024 * 1024 // NODE_BYTES
    sweep_start = time.perf_counter()

    for scene_cfg, model_cfgs in agents_by_scene():
        if args.scene and scene_cfg.tag != args.scene:
            continue

        json_path = Scene.save_dir(scene_cfg) / "reachability.json"
        seen = load_json(json_path)
        keep = seen if seen and seen.get("seed") == args.seed else None
        results = [
            Result(solved=bool(s), capped=bool(c))
            for s, c in zip(
                (keep or {}).get("episode_solved", []),
                (keep or {}).get("episode_capped", []),
            )
        ]
        skip = len(results)
        if skip >= args.episodes and not (keep or {}).get("partial", True):
            logger.info(f"{scene_cfg.tag}: already done ({skip} episodes)")
            continue

        while available_gb() < args.min_free_gb:
            logger.info(f"{available_gb():.1f} GB free < {args.min_free_gb}, waiting")
            time.sleep(30)

        heca = build_heca(scene_cfg, model_cfgs, args)
        scene, graph = heca.scene, heca.graph
        horizon = scene.cfg.max_steps
        labels = sorted(scene.entities)
        _reach, max_reach = option_reach(graph)
        started = time.perf_counter()
        logger.info(
            f"{scene_cfg.tag}: {args.episodes} episodes, horizon {horizon} options, "
            f"{len(graph.ns_option.items)} options, seed {args.seed}"
            + (f", resuming after {skip}" if skip else "")
        )

        for episode in range(1, args.episodes + 1):
            # seed the episode itself, so a resumed run reproduces the rest
            np.random.seed(args.seed + episode)
            scene.seed = args.seed + episode
            (x, _), (y, _) = scene.sample_task()
            if episode <= skip:
                continue  # already measured in an earlier run

            if scene.success():
                result = Result(True, False)  # already solved before any option
            else:
                result = search(
                    scene,
                    graph,
                    x,
                    y,
                    labels,
                    horizon,
                    args.max_nodes,
                    max_frontier,
                    max_reach,
                )
            results.append(result)
            summary = write_summary(
                json_path,
                results,
                scene_cfg.tag,
                args,
                horizon,
                time.perf_counter() - started,
                partial=True,
            )
            if episode % PROGRESS_EVERY == 0 or episode == args.episodes:
                logger.info(
                    f"  {scene_cfg.tag} {episode:>3}/{args.episodes}  "
                    f"solved {summary['solved']}"
                )

        summary = write_summary(
            json_path,
            results,
            scene_cfg.tag,
            args,
            horizon,
            time.perf_counter() - started,
            partial=False,
        )
        logger.info(
            f"{scene_cfg.tag}: solvable {summary['solved']}/{summary['episodes']} = "
            f"{100 * summary['reachable']:.1f}%"
            + (
                f" ({summary['capped']} capped, not proven)"
                if summary["capped"]
                else ""
            )
            + f" -> {json_path} ({fmt_duration(time.perf_counter() - started)})"
        )

        del heca, scene, graph
        clear_caches()
        gc.collect()

    logger.info(f"sweep took {fmt_duration(time.perf_counter() - sweep_start)}")


if __name__ == "__main__":
    main()
