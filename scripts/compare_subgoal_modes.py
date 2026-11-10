"""Compare the two chain-configuration constructions through the real pipeline.

    python -m scripts.compare_subgoal_modes --scene scene0

The construction is chosen by ``Entity.best_sample(mahalanobis=...)``; this script
forces the flag so both paths go through the unmodified ``Condition.make_subgoal``.
"""

import argparse
import itertools
from pathlib import Path

import joblib
import numpy as np

import heca.data.entity as entity_module
from heca.scenes.scene import Scene
from scripts.common.args import add_scene_argument
from scripts.common.scenes import experts_for_scene, scene_config

_ORIGINAL = entity_module.Entity.best_sample


def force_mode(mode: str):
    def best_sample(self, up1, up2, mahalanobis=True, eps=1e-15):
        return _ORIGINAL(self, up1, up2, mahalanobis=mode == "mahalanobis", eps=eps)

    entity_module.Entity.best_sample = best_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    args = parser.parse_args()
    scene = Scene.get(scene_config(args.scene))
    root = Path(scene.save_dir(scene.cfg)) / "experts"
    tags = [c.tag for c in experts_for_scene(args.scene)]
    conds = {t: joblib.load(root / t / "conditions-gt.joblib") for t in tags}
    ordered = list(itertools.product(tags, repeat=2))
    unordered = list(itertools.combinations(tags, 2))

    values = {}
    for mode in ("product", "mahalanobis"):
        force_mode(mode)
        chains = configs = 0
        emitted = refused = accepted = 0
        for a, b in ordered:
            subgoal = conds[b].pre.make_subgoal(conds[a].post)
            if not subgoal:
                continue
            chains += 1
            for key, value in subgoal.items():
                configs += 1
                values[(mode, a, b, key)] = np.asarray(value, float)
        for label in scene.entities:
            for a, b in unordered:
                for sa, sb in itertools.product(("pre", "post"), repeat=2):
                    ca, cb = getattr(conds[a], sa), getattr(conds[b], sb)
                    if label not in ca.models or label not in cb.models:
                        continue
                    subgoal = ca.make_subgoal(cb)
                    if not subgoal or label not in subgoal:
                        refused += 1
                        continue
                    emitted += 1
                    ent = scene.entities[label]
                    pose = ent.model_value(subgoal[label])
                    accepted += bool(
                        ent.score_prepared(
                            pose,
                            ent.scoring_parameters(ca.models[label].get_parameters().copy()),
                            rotation=True,
                        )
                        and ent.score_prepared(
                            pose,
                            ent.scoring_parameters(cb.models[label].get_parameters().copy()),
                            rotation=True,
                        )
                    )
        print(f"{mode:12s} chains {chains:3d}  configurations {configs:3d}  |  "
              f"sweep emitted {emitted:3d} refused {refused:3d} accepted-by-both {accepted:3d}")

    entity_module.Entity.best_sample = _ORIGINAL
    shared = [
        k[1:]
        for k in {(m, a, b, key) for (m, a, b, key) in values if m == "product"}
        if ("mahalanobis", *k[1:]) in values
    ]
    diffs = np.asarray([
        np.linalg.norm(values[("product", *k)][:3] - values[("mahalanobis", *k)][:3])
        for k in shared
    ])
    print(f"shared configurations {len(diffs)}: shift mean {diffs.mean():.3f} m, "
          f"max {diffs.max():.3f} m, differing {int((diffs > 1e-6).sum())}")


if __name__ == "__main__":
    main()
