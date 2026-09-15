"""Why is the option gate empty? Audit every gate row test on sampled tasks.

Replays what ``OptionNodes._gated`` tests (same rows, same ``Condition.test``),
attributes each failure to

* ``radius``  - outside the joint chi-square radius of the condition,
* ``dim``     - one pose dimension exceeds the per-dimension cap,
* ``state``   - the categorical state has probability ~0 under that component,

and reports which entity/mode/dimension is responsible. It also checks
sample/test consistency: draws from a condition must pass that condition's test.

Usage:
    PYG_HOME=/tmp/pyg_cache MPLCONFIGDIR=/tmp/mpl \
        python scripts/_diag_gate_rows.py scene4 200
"""

import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")

import numpy as np

from heca.data.entity import Entity, _chi_sqrt
from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph, SubgoalMode
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.graphs.nodes.node import EntityNode
from heca.graphs.roles import ENMode, ENRole
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config, find_scene_models

SAMPLES = 200
AXES = ("x", "y", "z")


def dim_name(dim: int) -> str:
    if dim < 3:
        return AXES[dim]
    if dim < 6:
        return f"rot{dim - 3}"
    return f"extra{dim}"


def row_report(node: EntityNode):
    """(passed, reason, z/chi, worst|z|, worst dim, state prob) for one row."""
    ent = node.con.entities[node.entity]
    up = node.con.models[node.entity].get_parameters().copy()  # type: ignore
    p = ent.secure_mix_parameters(up, add_variance=True)
    sample = ent.model_value(node.data.value)
    pose, state = sample[:-1], int(sample[-1])
    best_k, z, zd = ent._best_component(pose, p)
    chi = _chi_sqrt(ent.cfg.z_quantile_joint, ent.pose_dof)
    worst = float(np.max(np.abs(zd)))
    dim = int(np.argmax(np.abs(zd)))
    prob = float(p["measurement"]["state"]["pis"][best_k][state])
    if z > chi:
        return False, "radius", z / chi, worst, dim, prob
    if worst > ent._z_dim_sigma:
        return False, "dim", z / chi, worst, dim, prob
    if prob <= 1e-6:
        return False, "state", z / chi, worst, dim, prob
    return True, "", z / chi, worst, dim, prob


def audit(scene_tag: str, tasks: int) -> None:
    cfgs = find_scene_models(scene_tag)
    for cfg in cfgs:
        expert = ExpertModel.get(cfg, auto_load=False)
        expert.use_gt(True)
        expert.load()
        expert.virtual()
    graph = Graph.generate(list(cfgs), smode=SubgoalMode.BOTH)
    scene = Scene.get(find_scene_config(scene_tag), auto_load=False)
    print(
        f"\n=== {scene_tag}: {len(cfgs)} experts, {len(graph.ns_option.items)} options, "
        f"{tasks} sampled tasks ===",
        flush=True,
    )

    feasible: list[int] = []
    by_mode: dict[ENMode, list[bool]] = {m: [] for m in ENMode}
    by_role: dict[ENRole, list[bool]] = {r: [] for r in ENRole}
    reasons: Counter = Counter()
    dims: Counter = Counter()
    empty_tasks: list[list[str]] = []

    for _ in range(tasks):
        (x0, _), (y0, _) = scene.sample_task()
        graph.build(x0, y0, 0.0)  # budget 0.0, as at the first tick
        gated = graph.ns_option.gates(graph.ns_entity)
        feasible.append(int((gated == 0).sum()))

        first_failures: list[str] = []
        for option in graph.ns_option.items:
            failed_here = False
            rows: dict[str, EntityNode] = {}
            for key in option.sources[EntityNodes.type]:
                post = graph.ns_entity.get_by_key(key)
                assert isinstance(post, EntityNode)
                rows[key] = post
                for pre_key in post.sources.get(EntityNodes.type, ()):
                    pre = graph.ns_entity.get_by_key(pre_key)
                    assert isinstance(pre, EntityNode)
                    rows[pre_key] = pre
            for key in rows:
                node = rows[key]
                ok, reason, rel_z, worst, dim, prob = row_report(node)
                by_mode[node.mode].append(ok)
                by_role[node.role].append(ok)
                if ok:
                    continue
                reasons[reason] += 1
                dims[(Entity.TYPE_NAMES[node.type_id], dim_name(dim), node.role.name)] += 1
                if not failed_here:
                    failed_here = True
                    first_failures.append(
                        f"      {option.model.tag:<26} {node.entity:<8} "
                        f"role={node.role.name:<5} mode={node.mode.name:<7} "
                        f"{reason:<6} z/chi={rel_z:5.2f} worst|z|={worst:5.2f} "
                        f"dim={dim_name(dim):<6} p_state={prob:.1e}"
                    )
        if feasible[-1] == 0:
            empty_tasks.append(first_failures)

    counts = np.array(feasible)
    print(
        f"feasible options/task: min={counts.min()} median={int(np.median(counts))} "
        f"max={counts.max()} | empty-gate tasks: {int((counts == 0).sum())}/{tasks} "
        f"({(counts == 0).mean():.1%})"
    )
    for mode in ENMode:
        tests = by_mode[mode]
        if tests:
            print(
                f"  rows mode={mode.name:<8} n={len(tests):6d} "
                f"fail={sum(not t for t in tests) / len(tests):6.1%}"
            )
    for role in ENRole:
        tests = by_role[role]
        if tests:
            print(
                f"  rows role={role.name:<6} n={len(tests):6d} "
                f"fail={sum(not t for t in tests) / len(tests):6.1%}"
            )
    print(f"  failure reasons: {dict(reasons)}")
    print("  failing dimension (entity type, dim, role):")
    for key, count in dims.most_common(8):
        print(f"    {key}: {count}")

    for i, failures in enumerate(empty_tasks[:2]):
        print(f"  empty-gate task {i + 1}: first failing row per option")
        print("\n".join(failures))

    conditions: dict = {}
    for node in graph.ns_entity.items:
        assert isinstance(node, EntityNode)
        if node.con is not None:
            conditions.setdefault(node.con.label, node.con)
    print("  sample/test consistency (draws from a condition, tested on it):")
    for label, con in list(conditions.items())[:6]:
        rates, rel = [], []
        for entity in list(con.models):
            for _ in range(SAMPLES):
                probe = EntityNode(
                    entity=entity,
                    type_id=0,
                    data=con.sample(entity),
                    n_states=1,
                    mode=ENMode.SAMPLE,
                    role=ENRole.NONE,
                    con=con,
                    sources={},
                )
                ok, _, rel_z, _, _, _ = row_report(probe)
                rates.append(ok)
                rel.append(rel_z)
        print(
            f"    {label:<26} pass={np.mean(rates):6.1%} z/chi: "
            f"median={np.median(rel):5.2f} p90={np.percentile(rel, 90):5.2f} "
            f"max={np.max(rel):6.2f}"
        )


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "scene4"
    n_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    audit(tag, n_tasks)
