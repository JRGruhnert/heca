import h5py
import numpy as np
from heca.experts.expert import ExpertModel
from heca.scenes.scene import Scene
from scripts.common.scenes import experts_for_scene, scene_config

scene = Scene.get(scene_config("scene0"))
cfg = next(c for c in experts_for_scene("scene0") if c.tag == "window0_b_a")
model = ExpertModel.get(cfg, auto_load=False)
model.use_gt(False)
model.use_states = False
model.load()

print("live camera translation over 4 resets:")
for _ in range(4):
    (_, xf), _ = scene.sample_task(with_scenes=False)
    print("  ", np.round(np.asarray(xf.extr)[:3, 3], 3))

path = model.load_dir(cfg) / "demos.h5"
with h5py.File(path, "r") as f:
    rows = np.linspace(0, len(f["rgb"]) - 1, 4).astype(int)
    print("stored camera translation over 4 rows:")
    for row in rows:
        print("  ", np.round(np.asarray(f["extrinsics"][row])[:3, 3], 3))
