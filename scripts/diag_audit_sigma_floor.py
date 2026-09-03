"""Before/after acceptance audit for the per-dimension pose noise added in
``Entity.secure_mix_parameters`` (src/heca/data/entity.py).

Noise OFF (monkeypatched zeros) vs noise ON (config pos/rot/ext_sigma).

Run: python scripts/diag_audit_sigma_floor.py
"""

import numpy as np

from heca.data.entity import Entity
from heca.experts.expert import ExpertModel
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config, find_scene_models


def acceptance(tag, skill, starts):
    cfg = next(ac for ac in find_scene_models(tag) if ac.tag == skill)
    a = ExpertModel.get(cfg, auto_load=False)
    a.use_gt(True)
    a.load()
    con = a.conditions
    ok_all, per_ent = 0, {}
    for x in starts:
        ok = True
        for label, ent in a.entities.items():
            good = ent.score_single(
                x[label].value, con.pre.models[label].get_parameters()
            )
            per_ent[label] = per_ent.get(label, 0) + int(good)
            ok = ok and good
        ok_all += int(ok)
    return ok_all, per_ent


def run(tag, n=60):
    scene = Scene.get(find_scene_config(tag), auto_load=False)
    starts = [scene.sample_task()[0][0] for _ in range(n)]
    scene.close()

    orig = Entity.pose_sigma_variance
    Entity.pose_sigma_variance = lambda self: np.zeros(
        int(self.measurement["pose"]["n_columns"])
    )  # noise OFF
    print(f"== {tag} ==")
    for skill in ("lid0_base_base", "lid0_base_box0"):
        off_all, off_ent = acceptance(tag, skill, starts)
        Entity.pose_sigma_variance = orig  # noise ON (config defaults)
        on_all, on_ent = acceptance(tag, skill, starts)
        Entity.pose_sigma_variance = lambda self: np.zeros(
            int(self.measurement["pose"]["n_columns"])
        )
        print(
            f"  {skill:<16} noise OFF: {off_all}/{n} {off_ent}  "
            f"| noise ON: {on_all}/{n} {on_ent}"
        )
    Entity.pose_sigma_variance = orig


for tag in ("scene2", "scene4"):
    run(tag)
