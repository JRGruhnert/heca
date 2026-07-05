import h5py
import numpy as np

from heca.agents.experts.expert import ExpertAgent
from heca.agents.experts.tapas import TapasAgent
from heca.scenes.ogbench.scene import OGBenchScene

configs: list[TapasAgent.Config] = [
    # TapasAgent.Config(
    #     folder="open_drawer",
    #     scene=OGBenchScene.Config(),
    #     use_gt=True,
    # ),
    # TapasAgent.Config(
    #     folder="close_drawer",
    #     scene=OGBenchScene.Config(),
    #     use_gt=True,
    # ),
    # TapasAgent.Config(
    #     folder="open_window",
    #     scene=OGBenchScene.Config(),
    #     use_gt=True,
    # ),
    # TapasAgent.Config(
    #     folder="close_window",
    #     scene=OGBenchScene.Config(),
    #     use_gt=True,
    # ),
    # TapasAgent.Config(
    #     folder="lock_left_button",
    #     scene=OGBenchScene.Config(),
    #     use_gt=True,
    # ),
    # TapasAgent.Config(
    #     folder="lock_right_button",
    #     scene=OGBenchScene.Config(),
    #     use_gt=True,
    # ),
    # TapasAgent.Config(
    #     folder="unlock_left_button",
    #     scene=OGBenchScene.Config(),
    #     use_gt=True,
    # ),
    # TapasAgent.Config(
    #     folder="unlock_right_button",
    #     scene=OGBenchScene.Config(),
    #     use_gt=True,
    # ),
    # TapasAgent.Config(
    #     folder="move_block",
    #     scene=OGBenchScene.Config(),
    #     use_gt=True,
    # ),
    TapasAgent.Config(
        tag="move_block_drawer",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
]

for cfg in configs:
    for name in ["demos", "demos_post"]:
        load_path = ExpertAgent.load_dir(cfg) / f"{name}.h5"
        save_path = ExpertAgent.load_dir(cfg) / f"{name}_new.h5"

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
