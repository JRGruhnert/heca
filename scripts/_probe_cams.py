import numpy as np
from heca.scenes.scene import Scene
from scripts.common.scenes import scene_config

scene = Scene.get(scene_config("scene0"))
scene.sample_task(with_scenes=False)          # bring the env up
env = getattr(scene, "env", None) or scene._env
model, data = env._model or env.model, env._data or env.data
print("cameras:", model.ncam)
for i in range(model.ncam):
    name = model.camera(i).name
    T = np.eye(4, dtype=np.float32)
    R = data.cam_xmat[i].reshape(3, 3)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ data.cam_xpos[i]
    print(f"  [{i}] {name!r:24s} xpos {np.round(data.cam_xpos[i], 3)} "
          f"T_cw t {np.round(T[:3, 3], 3)}")
print("current _camera:", getattr(env, "_camera", None),
      "resolved:", env._resolve_camera_id(None))
