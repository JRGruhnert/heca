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
path = model.load_dir(cfg) / "demos.h5"
KEYS = {"rgb", "depth", "mask", "extrinsics", "intrinsics", "demo"}

print("defaults      center", scene._meta_xyz_center, "scaler", scene._meta_xyz_scaler)
with h5py.File(path, "r") as f:
    row = 0
    extra = {k: f[k][row] for k in f.keys() if k not in KEYS}
    print("h5 meta       center", extra.get("meta_xyz_center"), "scaler", extra.get("meta_xyz_scaler"))
    obs, _ = scene.to_internal({k: f[k][row] for k in KEYS if k != "demo"}, extra)
    scene.to_td_image(obs)
    print("after demo    center", scene._meta_xyz_center, "scaler", scene._meta_xyz_scaler)

(_, xf), _ = scene.sample_task(with_scenes=False)
print("after live    center", scene._meta_xyz_center, "scaler", scene._meta_xyz_scaler)
env = getattr(scene, "env", None) or scene._env
ob, info = env.reset(options={"render_goal": True})
print("env info keys with meta:", [k for k in info if "meta" in k or "xyz" in k])
if "meta_xyz_center" in info:
    print("live meta     center", np.asarray(info["meta_xyz_center"]),
          "scaler", np.asarray(info["meta_xyz_scaler"]))
