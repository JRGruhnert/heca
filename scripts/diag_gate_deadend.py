"""Hunt for empty option gates inside episodes and dump why they happened.

Rolls out episodes like ``Heca.act`` but picks uniformly among the feasible
options. Whenever the gate is empty, every option is printed with its failing
rows (precondition at the current state vs postcondition on the goal), so the
dead end is attributed instead of guessed.

Usage: PYG_HOME=/tmp/pyg_cache MPLCONFIGDIR=/tmp/mpl \
           python scripts/diag_gate_deadend.py scene4 60
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg")
import numpy as np
from heca.data.entity import _chi_sqrt
from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph, SubgoalMode
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.graphs.nodes.node import EntityNode
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config, find_scene_models

MAX_STEPS = 32

def row_state(node):
    ent = node.con.entities[node.entity]
    up = node.con.models[node.entity].get_parameters().copy()
    p = ent.secure_mix_parameters(up, add_variance=True)
    sample = ent.model_value(node.data.value)
    pose, state = sample[:-1], int(sample[-1])
    best_k, z, zd = ent._best_component(pose, p)
    chi = _chi_sqrt(ent.cfg.z_quantile_joint, ent.pose_dof)
    prob = float(p["measurement"]["state"]["pis"][best_k][state])
    ok = z <= chi and bool(np.all(ent.group_squared(zd) <= ent._z_dim_sigma**2)) and prob > 1e-6
    return bool(ok), z / chi, float(np.max(np.abs(zd))), prob

def dump(graph, out):
    gated = graph.ns_option.gates(graph.ns_entity)
    out.write(f"options={len(graph.ns_option.items)} gated_vector={gated.tolist()}\n")
    out.write(f"gated_sum={int(gated.sum())} usable={int((gated == 0).sum())}\n")
    for i, option in enumerate(graph.ns_option.items):
        key = graph.ns_option.key_at(i)
        fails = []
        rows = {}
        for src in option.sources[EntityNodes.type]:
            post = graph.ns_entity.get_by_key(src)
            rows[src] = post
            for pre_src in post.sources.get(EntityNodes.type, ()):
                rows[pre_src] = graph.ns_entity.get_by_key(pre_src)
        for src in sorted(rows):
            node = rows[src]
            ok, rel, worst, prob = row_state(node)
            if not ok:
                fails.append(f"{node.entity}/{node.mode.name}/{node.role.name} z/chi={rel:.2f} w={worst:.2f} p={prob:.0e}")
        out.write(f"[{i:2d}] {key:<34} gated={gated[i]:.0f} _gated={graph.ns_option._gated(option, graph.ns_entity)} fails={fails}\n")

def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "scene4"
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    cfgs = find_scene_models(tag)
    for cfg in cfgs:
        e = ExpertModel.get(cfg, auto_load=False); e.use_gt(True); e.load(); e.virtual()
    graph = Graph.generate(list(cfgs), smode=SubgoalMode.BOTH)
    scene = Scene.get(find_scene_config(tag), auto_load=False)
    rng = np.random.RandomState(0)
    with open("logs/gate_deadend_dump.txt", "w") as out:
        for ep in range(episodes):
            (x, _), (y, _) = scene.sample_task()
            budget = 0.0
            for step in range(1, MAX_STEPS + 1):
                graph.build(x, y, budget)
                gated = graph.ns_option.gates(graph.ns_entity)
                usable = int((gated == 0).sum())
                if usable == 0:
                    out.write(f"DEAD END ep{ep} step{step} budget={budget:.2f}\n")
                    dump(graph, out)
                    print(f"dead end at ep{ep} step{step}; dumped to logs/gate_deadend_dump.txt")
                    return
                if usable < 3:
                    out.write(f"near dead end ep{ep} step{step} budget={budget:.2f} usable={usable}\n")
                    dump(graph, out)
                choice = int(rng.choice(np.flatnonzero(gated.numpy() == 0)))
                a, s = graph.select(choice)
                z, fb = ExpertModel.get(a).act(x, s)
                x, budget = z, fb.budget
                if fb.end:
                    break
        print(f"no dead end in {episodes} episodes")

main()
