"""Pre-condition reliability audit per scene / skill / entity.

For every agent of a scene, sample N episodes from the same env the trainer
uses and measure how often the fitted pre-condition accepts the episode start:
  - overall skill validity (all entities of the skill pass),
  - per-entity acceptance,
  - per-entity failure cause: state-gate vs pose-gate.

Uses the real ``Entity.score_single`` (incl. the pose-noise added in
``secure_mix_parameters``), so results reflect the current conditions.

Run: python scripts/diag_reliability.py --scene scene2 [--episodes 60]
"""

import argparse
import math

import numpy as np
from scipy.stats import chi2, norm

from heca.experts.expert import ExpertModel
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config, find_scene_models


def classify(ent, sample, up):
    """Returns (ok, ok_pose, ok_state) replicating Entity.score_single.

    Uses ``ent.secure_mix_parameters`` so the pose-noise is included.
    """
    p = ent.secure_mix_parameters(up)
    pose = ent.model_value(sample)[:-1]
    state = int(ent.model_value(sample)[-1])
    pis = p["measurement"]["state"]["pis"]
    best_k, z, zd = ent._best_component(pose, p)
    chi_sqrt = math.sqrt(chi2.ppf(ent.cfg.z_quantile_joint, len(pose)))
    cap = float(norm.ppf(0.5 + ent.cfg.z_quantile_dim / 2.0))
    ok_pose = z <= chi_sqrt and bool(np.all(zd <= cap))
    ok_state = state == int(np.argmax(pis[best_k]))
    return ok_pose and ok_state, ok_pose, ok_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--episodes", type=int, default=60)
    args = parser.parse_args()

    scene_cfg = find_scene_config(args.scene)
    scene = Scene.get(scene_cfg, auto_load=False)
    agents = []
    for ac in find_scene_models(args.scene):
        a = ExpertModel.get(ac, auto_load=False)
        a.use_gt(True)
        a.load()
        agents.append(a)

    n = args.episodes
    starts = [scene.sample_task()[0][0] for _ in range(n)]
    scene.close()

    print(f"== {args.scene}: pre-condition reliability over {n} sampled starts ==")
    for a in agents:
        con = a.conditions
        ok_all, per_ent = 0, {}
        for x in starts:
            ok = True
            for label, ent in a.entities.items():
                ok_p, ok_pose, ok_state = classify(
                    ent, x[label].value,
                    a.conditions.pre.models[label].get_parameters(),
                )
                per_ent.setdefault(label, [0, 0, 0])  # [ok, pose-fail, state-fail]
                per_ent[label][0] += int(ok_p)
                if not ok_pose:
                    per_ent[label][1] += 1
                if not ok_state:
                    per_ent[label][2] += 1
                ok = ok and ok_p
            ok_all += int(ok)
        role = lambda l: "T" if l in con.target_entities else "A"
        parts = "  ".join(
            f"{l}({role(l)}):{v[0]}/{n}[p{v[1]}/s{v[2]}]"
            for l, v in sorted(per_ent.items())
        )
        print(f"  {a.cfg.tag:<20} all:{ok_all:>3}/{n}   {parts}")
    print()


if __name__ == "__main__":
    main()
