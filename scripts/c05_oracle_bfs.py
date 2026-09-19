import argparse
import heapq
import itertools
import json
import time
from collections import deque
from dataclasses import dataclass, field, replace
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")

from heca.data.data import DCScene
from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph, SubgoalMode
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.graphs.roles import ENMode
from heca.scenes.scene import Scene
from heca.misc import logger

from scripts.common.helper import fmt_duration
from scripts.common.scenes import find_scene_models

PLAN_COLUMNS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)  # "solved within k options"


def snapshot(scene: Scene) -> tuple:
    """qpos/qvel plus the per-object state the env keeps outside the sim."""
    env = scene.env
    objects = []
    for obj in env.objects:
        rec: dict[str, Any] = {}
        for attr in ("_cur_state", "_target_button_states"):
            value = getattr(obj, attr, None)
            rec[attr] = value.copy() if value is not None else None
        rec["_target_val"] = getattr(obj, "_target_val", None)
        objects.append((obj, rec))
    return env._data.qpos.copy(), env._data.qvel.copy(), objects


def restore(scene: Scene, snap: tuple, depth: int) -> None:
    env = scene.env
    qpos, qvel, objects = snap
    env._data.qpos[:] = qpos
    env._data.qvel[:] = qvel
    for obj, rec in objects:
        for attr, value in rec.items():
            if value is None:
                continue
            if attr == "_target_val":
                setattr(obj, attr, value)
            else:
                getattr(obj, attr)[:] = value
    env._apply_button_states()
    # option counting lives on the scene, not in the sim, so it has to follow
    scene.current_step = depth


def goals(scene: Scene) -> dict[str, bool]:
    """The env's per-object success bookkeeping for the current state."""
    return {name: bool(val) for val, name in scene.env._compute_successes()}


def goal_reached(scene: Scene) -> bool:
    """The env's own success criterion for the state it currently holds."""
    env = scene.env
    return bool(env._evaluate_success(env._compute_successes()))


def state_key(x: DCScene, labels: list[str], digits: int) -> tuple:
    return tuple(
        np.round(np.asarray(x.get(label).value, dtype=float), digits).tobytes()
        for label in labels
    )


# --- search ----------------------------------------------------------------


@dataclass
class Node:
    x: DCScene
    budget: float
    depth: int
    snap: tuple
    path: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class Result:
    solved: bool
    depth: int | None
    path: tuple[str, ...]
    nodes: int
    dead_ends: int
    capped: bool
    greedy_depth: int | None = None
    greedy_path: tuple[str, ...] = ()
    start_unsatisfied: int = 0
    seconds: float = 0.0  # wall time this episode's search took


def option_reach(graph: Graph) -> dict[int, int]:
    """How many objects an option can complete in one application.

    The env only teleports the entities of the subgoal it is handed, i.e. the
    rows that carry a target mode; anchors keep their current value. That bound
    is what makes the A* heuristic below admissible.
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
    return reach


def heuristic(unsatisfied: int, reach: int) -> int:
    """Least number of further options: each one completes at most ``reach``."""
    return -(-unsatisfied // reach)


def feasible_options(graph: Graph, x: DCScene, y: DCScene, budget: float):
    """The option indices that pass the gate for this state."""
    data = graph.build(x, y, budget)
    gated = data.option.gated
    return np.flatnonzero(gated.numpy() == 0).tolist()


def greedy_chain(
    scene: Scene,
    graph: Graph,
    x: DCScene,
    y: DCScene,
    max_steps: int,
) -> tuple[int, tuple[str, ...]] | None:
    cur, budget = x, 0.0
    path: list[str] = []
    for step in range(1, max_steps + 1):
        data = graph.build(cur, y, budget)
        options = np.flatnonzero(data.option.gated.numpy() == 0).tolist()
        if not options:
            return None
        base = snapshot(scene)
        depth = scene.current_step
        best: tuple[int, int, DCScene, Any, tuple] | None = None
        for index in options:
            restore(scene, base, depth)
            agent_cfg, subgoal = graph.select(index)
            z, fb = ExpertModel.get(agent_cfg).act(cur, subgoal, gated=False)
            if goal_reached(scene):
                return step, tuple(path + [graph.ns_option.key_at(index)])
            if best is None or sum(goals(scene).values()) > best[0]:
                best = (sum(goals(scene).values()), index, z, fb, snapshot(scene))
        assert best is not None
        _, index, z, fb, snap = best
        restore(scene, snap, step)
        path.append(graph.ns_option.key_at(index))
        cur, budget = z, fb.budget
    return None


def bfs(
    scene: Scene,
    graph: Graph,
    x: DCScene,
    y: DCScene,
    labels: list[str],
    max_steps: int,
    max_nodes: int,
    branching: int,
    digits: int,
    strategy: str,
    reach: dict[int, int],
) -> Result:
    """Shortest option sequence from ``x`` to a success state.

    ``strategy="bfs"`` expands strictly level by level; ``strategy="astar"`` orders
    the frontier by ``g + h`` with the admissible ``heuristic`` above. Both return
    the same minimum, A* just needs far fewer expansions.
    """
    start = Node(x=x, budget=0.0, depth=0, snap=snapshot(scene))
    start_unsatisfied = sum(1 for ok in goals(scene).values() if not ok)
    # a greedy oracle is cheap and gives an upper bound on the optimal length
    greedy = greedy_chain(scene, graph, x, y, max_steps)
    greedy_depth, greedy_path = (greedy if greedy else (None, ()))
    if greedy_depth == 1:  # nothing can beat one option, no search needed
        return Result(
            True, 1, greedy_path, 0, 0, False, greedy_depth, greedy_path, start_unsatisfied
        )

    restore(scene, start.snap, 0)
    n_objects = len(goals(scene))
    max_reach = max(reach.values()) if reach else 1
    ties = itertools.count()
    queue: deque[Node] = deque([start])
    heap: list[tuple[int, int, int, Node]] = [
        (heuristic(start_unsatisfied, max_reach), 0, next(ties), start)
    ]
    best_g: dict[tuple, int] = {state_key(x, labels, digits): 0}
    nodes = 0
    dead_ends = 0
    capped = False

    while queue if strategy == "bfs" else heap:
        if strategy == "bfs":
            node = queue.popleft()
        else:
            _, popped_g, _, node = heapq.heappop(heap)
            if popped_g > best_g.get(state_key(node.x, labels, digits), popped_g):
                continue  # superseded by a cheaper route to the same state
        if node.depth >= max_steps:
            continue
        if nodes >= max_nodes:
            capped = True
            break

        restore(scene, node.snap, node.depth)
        options = feasible_options(graph, node.x, y, node.budget)
        if not options:
            dead_ends += 1
            continue
        if branching:
            options = options[:branching]

        children: list[tuple[int, Node]] = []
        for index in options:
            restore(scene, node.snap, node.depth)
            agent_cfg, subgoal = graph.select(index)
            z, fb = ExpertModel.get(agent_cfg).act(node.x, subgoal, gated=False)
            nodes += 1

            path = node.path + (graph.ns_option.key_at(index),)
            if goal_reached(scene):
                result = Result(True, node.depth + 1, path, nodes, dead_ends, capped)
                result.greedy_depth, result.greedy_path = greedy_depth, greedy_path
                result.start_unsatisfied = start_unsatisfied
                return result
            if fb.end:  # episode over without success: nothing to expand
                continue

            depth = node.depth + 1
            key = state_key(z, labels, digits)
            if best_g.get(key, depth + 1) <= depth:
                continue  # this state was already reached as cheaply
            best_g[key] = depth
            children.append(
                (
                    sum(goals(scene).values()),
                    Node(x=z, budget=fb.budget, depth=depth, snap=snapshot(scene), path=path),
                )
            )

        if strategy == "bfs":
            # ordering inside a level does not break optimality
            children.sort(key=lambda item: -item[0])
            queue.extend(child for _, child in children)
        else:
            for satisfied, child in children:
                heapq.heappush(
                    heap,
                    (
                        child.depth + heuristic(n_objects - satisfied, max_reach),
                        child.depth,
                        next(ties),
                        child,
                    ),
                )

    result = Result(False, None, (), nodes, dead_ends, capped)
    result.greedy_depth, result.greedy_path = greedy_depth, greedy_path
    result.start_unsatisfied = start_unsatisfied
    return result


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, the honest one for counts near 0 or 1."""
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(results: list[Result], max_steps: int) -> dict:
    solved = [r for r in results if r.solved]
    depths = np.array([r.depth for r in solved], dtype=int)
    greedy = np.array(
        [r.greedy_depth for r in results if r.greedy_depth is not None], dtype=int
    )
    unsatisfied = np.array([r.start_unsatisfied for r in results], dtype=int)
    seconds = np.array([r.seconds for r in results], dtype=float)
    nodes = np.array([r.nodes for r in results], dtype=int)
    lo, hi = wilson(len(solved), len(results))
    columns = sorted({*[k for k in PLAN_COLUMNS if k <= max_steps], max_steps})
    within = {
        k: float(np.mean([r.solved and r.depth <= k for r in results])) for k in columns
    }
    summary_columns = columns
    return {
        "episodes": len(results),
        "solved": len(solved),
        "reachable": len(solved) / len(results) if results else 0.0,
        "reachable_ci95": [lo, hi],
        "max_steps": max_steps,
        "depths": [int(r.depth) if r.solved else None for r in results],
        "depth_min": int(depths.min()) if depths.size else None,
        "depth_median": float(np.median(depths)) if depths.size else None,
        "depth_mean": float(depths.mean()) if depths.size else None,
        "depth_max": int(depths.max()) if depths.size else None,
        "solved_within": {str(k): v for k, v in within.items()},
        "solved_within_columns": summary_columns,
        "greedy_solved": int(greedy.size),
        "greedy_depth_median": float(np.median(greedy)) if greedy.size else None,
        "greedy_depth_mean": float(greedy.mean()) if greedy.size else None,
        "greedy_depth_max": int(greedy.max()) if greedy.size else None,
        "start_unsatisfied_median": (
            float(np.median(unsatisfied)) if unsatisfied.size else None
        ),
        "start_unsatisfied_max": int(unsatisfied.max()) if unsatisfied.size else 0,
        "capped": sum(r.capped for r in results),
        "nodes": sum(r.nodes for r in results),
        "dead_ends": sum(r.dead_ends for r in results),
        # cost of finding the reported path, per episode: wall time and search
        # expansions (the search stops at the first solution, so this is the
        # time to find that minimal path)
        "episode_seconds": [round(s, 2) for s in seconds],
        "episode_nodes": [int(n) for n in nodes],
        "seconds_median": float(np.median(seconds)) if seconds.size else None,
        "seconds_mean": float(seconds.mean()) if seconds.size else None,
        "seconds_max": float(seconds.max()) if seconds.size else None,
        "nodes_median": float(np.median(nodes)) if nodes.size else None,
        "nodes_max": int(nodes.max()) if nodes.size else 0,
        "ms_per_expansion": float(
            1000 * seconds.sum() / max(nodes.sum(), 1)
        ),
        "paths": {str(i): list(r.path) for i, r in enumerate(results) if r.solved},
    }


def report(scene_tag: str, summary: dict, elapsed: float) -> None:
    total = summary["episodes"]
    lo, hi = summary["reachable_ci95"]
    logger.info(
        f"{scene_tag}: {total} episodes, horizon {summary['max_steps']} options"
    )
    logger.info(
        f"  reachable: {summary['solved']}/{total} = "
        f"{100 * summary['reachable']:.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)"
    )
    if summary["solved"]:
        logger.info(
            f"  optimal options: min {summary['depth_min']}, "
            f"median {summary['depth_median']:.0f}, mean {summary['depth_mean']:.1f}, "
            f"max {summary['depth_max']}"
        )
    logger.info(
        f"  objects to change in the start state: median "
        f"{summary['start_unsatisfied_median']:.0f}, max "
        f"{summary['start_unsatisfied_max']}"
    )
    if summary["greedy_solved"]:
        logger.info(
            f"  greedy oracle: {summary['greedy_solved']}/{total} solved, "
            f"median {summary['greedy_depth_median']:.0f} options, "
            f"mean {summary['greedy_depth_mean']:.1f}, max {summary['greedy_depth_max']}"
            + (
                f" (BFS optimum is {summary['depth_median']:.0f} median)"
                if summary["solved"]
                else ""
            )
        )
    curve = " / ".join(f"{100 * v:.0f}%" for v in summary["solved_within"].values())
    logger.info(
        f"  solved within {'/'.join(str(k) for k in summary['solved_within_columns'])} "
        f"options: {curve}"
    )
    logger.info(
        f"  cost to find the path: median {summary['seconds_median']:.1f}s "
        f"({summary['nodes_median']:.0f} expansions), mean "
        f"{summary['seconds_mean']:.1f}s, max {summary['seconds_max']:.1f}s "
        f"({summary['nodes_max']} expansions), "
        f"{summary['ms_per_expansion']:.1f} ms/expansion"
    )
    logger.info(
        f"  search: {summary['nodes'] / max(total, 1):.0f} expansions/episode, "
        f"{summary['dead_ends']} states with no feasible option, "
        f"{summary['capped']} episodes hit the node cap"
        + ("" if not summary["capped"] else " (those are not proven unreachable)")
    )
    deepest = max(
        (d for d in summary["depths"] if d is not None),
        default=None,
    )
    if deepest is not None:
        index = summary["depths"].index(deepest)
        logger.info(
            f"  longest optimal path ({deepest} options): "
            + " -> ".join(summary["paths"][str(index)])
        )
    logger.info(f"  took {fmt_duration(elapsed)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", default="scene0", help="scene tag, e.g. scene0")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="option horizon; 0 uses the scene's own max_steps",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=60000,
        help="search budget per episode (expansions of one option application, "
        "~3 ms each); episodes that hit it are reported as *not proven* "
        "unreachable, so raise it to settle those",
    )
    parser.add_argument(
        "--search",
        choices=("astar", "bfs"),
        default="astar",
        help="astar (default; admissible heuristic, same optimum as bfs) or bfs",
    )
    parser.add_argument(
        "--branching",
        type=int,
        default=0,
        help="options expanded per state, 0 = all feasible ones",
    )
    parser.add_argument(
        "--smode", type=SubgoalMode, choices=list(SubgoalMode), default=SubgoalMode.BOTH
    )
    parser.add_argument(
        "--digits", type=int, default=3, help="rounding used to deduplicate states"
    )
    parser.add_argument(
        "--rotation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="score conditions with orientation (the gate ignores it otherwise)",
    )
    parser.add_argument(
        "--gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use ground-truth observations for the conditions",
    )
    parser.add_argument(
        "--mode",
        default="randomized",
        help="scene env mode: 'randomized' (the training distribution) or 'task'",
    )
    parser.add_argument(
        "--out",
        default="plots/reachability",
        help="directory for the per-scene JSON summary (default plots/reachability)",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    cfgs = [
        replace(cfg, scene=replace(cfg.scene, mode=args.mode))
        for cfg in find_scene_models(args.scene)
    ]
    for cfg in cfgs:
        expert = ExpertModel.get(cfg, auto_load=False)
        expert.use_gt(args.gt)
        expert.load()
        expert.virtual()

    graph = Graph.generate(
        list(cfgs),
        smode=args.smode,
        use_rotation=args.rotation,
        position_jitter=0.0,
    )
    scene = Scene.get(cfgs[0].scene)
    # the scene is a singleton built when conf.scenes is imported (randomized),
    # so the mode has to be switched on the instance, not on a config copy
    scene.set_mode(args.mode)
    max_steps = args.max_steps or scene.cfg.max_steps
    labels = sorted(scene.entities)

    logger.info(
        f"{args.scene}: {len(graph.ns_option.items)} options, {len(cfgs)} experts, "
        f"mode={args.mode}, virtual mode, horizon {max_steps} options"
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.scene}.json"

    started = time.perf_counter()
    results: list[Result] = []

    def dump(partial: bool) -> None:
        summary = summarise(results, max_steps)
        summary.update(
            scene=args.scene,
            seed=args.seed,
            mode=args.mode,
            smode=args.smode.value,
            branching=args.branching,
            search=args.search,
            max_nodes=args.max_nodes,
            partial=partial,
            elapsed_s=round(time.perf_counter() - started, 1),
        )
        path.write_text(json.dumps(summary, indent=2))

    for episode in range(1, args.episodes + 1):
        if args.mode == "task":
            # cycle the benchmark's tasks instead of drawing them at random, so
            # the summary is per task and not per lucky draw
            scene.task_id = (episode - 1) % len(scene.tasks) + 1  # type: ignore[attr-defined]
        (x, _), (y, _) = scene.sample_task()
        if goal_reached(scene):
            results.append(Result(True, 0, (), 0, 0, False, 0, (), 0))
            dump(partial=True)
            logger.info(f"  {episode:>3}/{args.episodes}  already solved at the start")
            continue
        began = time.perf_counter()
        result = bfs(
            scene,
            graph,
            x,
            y,
            labels,
            max_steps,
            args.max_nodes,
            args.branching,
            args.digits,
            args.search,
            option_reach(graph),
        )
        result.seconds = time.perf_counter() - began
        results.append(result)
        dump(partial=True)
        if result.solved:
            logger.info(
                f"  {episode:>3}/{args.episodes}  solved in {result.depth} options "
                f"({result.nodes} expansions, {result.seconds:.1f}s)"
            )
        else:
            logger.info(
                f"  {episode:>3}/{args.episodes}  not reachable within {max_steps} "
                f"options ({result.nodes} expansions"
                + (", capped)" if result.capped else ", frontier exhausted)")
            )

    elapsed = time.perf_counter() - started
    dump(partial=False)
    summary = json.loads(path.read_text())
    report(args.scene, summary, elapsed)
    logger.info(f"summary written to {path}")


if __name__ == "__main__":
    main()
