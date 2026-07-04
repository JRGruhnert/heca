from heca.agents.agent import Agent
from heca.agents.experts.tapas import TapasAgent

# from heca.agents.heca import Heca
from heca.agents.heca import Heca
from heca.scenes.ogbench.scene import OGBenchScene

agents = [
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
    TapasAgent.Config(
        folder="unlock_left_button",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
    TapasAgent.Config(
        folder="unlock_right_button",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
    TapasAgent.Config(
        folder="move_block",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
    # TapasAgent.Config(
    #    folder="move_ee",
    #    scene=OGBenchScene.Config(),
    #    use_gt=True,
    # ),
]


heca_cfg = Heca.Config(
    subroot="heca_test",
    folder="test",
    agents=agents,
)

heca = Agent.get(heca_cfg, load=False)
heca.conditions
