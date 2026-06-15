from typing import cast
from ogbench.manipspace.envs.scene_env import ManipSpaceEnv

import gymnasium
import ogbench

from heca.agents.experts.expert import ExpertAgent
from heca.agents.experts.tapas import TapasAgent
from heca.environment.scenes.ogbench.scene import OGBenchScene

print("before")

env = gymnasium.make(
    "visual-scene-v0",
    ob_type="pixels",
)

print("after")

env = cast(
    ManipSpaceEnv,
    ogbench.make_env_and_datasets(
        dataset_name="visual-scene-play-v0",
        env_only=True,
        dataset_only=False,
        # ob_type=cfg.ob_type,
        # mode=cfg.mode,
        # visualize_info=cfg.visualize_info,
        # width=cfg.width,
        # height=cfg.height,
    ),
)

print("after2")


agent = ExpertAgent.get(
    TapasAgent.Config(
        folder="close_drawer",
        scene=OGBenchScene.Config(),
    ),
)

print("after3")
