import h5py
import numpy as np

from heca.agents.experts.expert import ExpertAgent
from heca.agents.experts.tapas import TapasAgent
from heca.environment.scenes.ogbench.scene import OGBenchScene

cfg = TapasAgent.Config(
    folder="lock_left_button",
    scene=OGBenchScene.Config(),
)

load_path = ExpertAgent.resolve(cfg) / "demos_post.h5"
save_path = ExpertAgent.resolve(cfg) / "demos_post_new.h5"

file = h5py.File(load_path, "r")

with h5py.File(load_path, "r") as f:
    data = {k: np.asarray(f[k]) for k in f.keys()}

# Flip 0 <-> 1 for privileged_button_0_state
data["privileged_button_0_pos_full"] = data["privileged_button_0_pos"]
data["privileged_button_1_pos_full"] = data["privileged_button_1_pos"]
# Save everything back
with h5py.File(save_path, "w") as f:
    for key, value in data.items():
        f.create_dataset(
            key,
            data=value,
            compression="gzip",
        )

print(f"Saved modified dataset to {save_path}")
