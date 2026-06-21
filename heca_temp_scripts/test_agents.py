from heca.agents.experts.tapas import TapasAgent
from heca.scenes.ogbench.scene import OGBenchScene
from heca.guis.tapas_agent_tester import AgentTester
import numpy as np

np.set_printoptions(
    linewidth=200,
    suppress=True,
)
cfg = AgentTester.Config(
    agents=[
        TapasAgent.Config(
            folder="open_drawer",
            scene=OGBenchScene.Config(),
            use_gt=True,
        ),
        TapasAgent.Config(
            folder="close_drawer",
            scene=OGBenchScene.Config(),
            use_gt=True,
        ),
        TapasAgent.Config(
            folder="open_window",
            scene=OGBenchScene.Config(),
            use_gt=True,
        ),
        TapasAgent.Config(
            folder="close_window",
            scene=OGBenchScene.Config(),
            use_gt=True,
        ),
        TapasAgent.Config(
            folder="lock_left_button",
            scene=OGBenchScene.Config(),
            use_gt=True,
        ),
        TapasAgent.Config(
            folder="lock_right_button",
            scene=OGBenchScene.Config(),
            use_gt=True,
        ),
    ],
    scene=OGBenchScene.Config(),
)
tester = AgentTester.get(cfg)
tester.run()
