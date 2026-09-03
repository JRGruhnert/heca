"""Conditional skill reliability: given a fully pre-valid start, does the
(virtual) execution of the skill place all its target entities on the subgoal?

For each skill over N sampled episode starts:
  V      = starts where ALL entities pass the pre-condition (skill VALID)
  S      = among those V, starts where every target entity of the skill ends up
           at its proposed subgoal value (graph semantics: y if inside the
           post-condition, else a post sample)

Run: python scripts/diag_skill_success.py --scene scene2 [--episodes 120]
"""

import argparse

import numpy as np

from heca.data.data import DCScene
from heca.experts.expert import ExpertModel
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config, find_scene_models
from scripts.diag_trainability import (
    agent_subgoal,
    env_restore,
    env_snapshot,
    mismatched,
    pre_valid,
)

POS_TOL = 0.05  # normalized (~5 mm)
ANG_TOL = 0.02  # rad


def target_ok(agent, z: DCScene, sub: DCScene) -> list[str]:
    """Target entities of `agent` whose z value != subgoal value (loose tol)."""
    bad = []
    for label in agent.conditions.target_entities:
        a = z.get(label).value
        b = sub.get(label).value
        kind = None
        # infer kind by value length / entity type id
        ent = agent.entities[label]
        t = ent.cfg.type_id  # 0 free, 1 static, 2 prismatic, 3 revolute
        if t == 1:  # static: state must match (pose fixed)
            if int(round(a[-1])) != int(round(b[-1])):
                bad.append(label)
        elif t == 3:  # revolute: angle via sin/cos extras
            ang_a = float(np.arctan2(a[6], a[7]))
            ang_b = float(np.arctan2(b[6], b[7]))
            if abs(ang_a - ang_b) > ANG_TOL:
                bad.append(label)
        elif t == 2:  # prismatic: extra value
            if abs(a[6] - b[6]) > 0.02:
                bad.append(label)
        else:  # free
            if np.linalg.norm(a[:3] - b[:3]) > POS_TOL:
                bad.append(label)
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--episodes", type=int, default=120)
    args = parser.parse_args()

    scene_cfg = find_scene_config(args.scene)
    scene = Scene.get(scene_cfg, auto_load=False)
    agents = []
    for ac in find_scene_models(args.scene):
        a = ExpertModel.get(ac, auto_load=False)
        a.use_gt(True)
        a.virtual()
        a.load()
        agents.append(a)

    n = args.episodes
    print(f"== {args.scene}: conditional skill success over {n} starts ==")
    print("   (V=pre-valid starts, S=successful executions among V)")
    for a in agents:
        V = 0
        S = 0
        fail_targets: dict[str, int] = {}
        for _ in range(n):
            (x, _), (y, _) = scene.sample_task()
            if not pre_valid(a, x):
                continue
            V += 1
            snap = env_snapshot(scene)
            try:
                sub = agent_subgoal(x, y, a, "graph")
                z, _ = a.act(x, sub)
                bad = target_ok(a, z, sub)
                if not bad:
                    S += 1
                else:
                    for b in bad:
                        fail_targets[b] = fail_targets.get(b, 0) + 1
            finally:
                env_restore(scene, snap)
        ratio = f"{S}/{V}" if V else "-"
        extra = ""
        if fail_targets:
            extra = "  fail-targets: " + ", ".join(
                f"{k}:{v}" for k, v in sorted(fail_targets.items())
            )
        print(f"  {a.cfg.tag:<20} V={V:>3}  success {ratio}{extra}")
    scene.close()


if __name__ == "__main__":
    main()
