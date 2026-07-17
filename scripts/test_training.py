from heca.agents.agent import Agent
from heca.agents.experts.tapas import TapasAgent

# from heca.agents.heca import Heca
from heca.agents.heca import Heca
from heca.conditions.evaluator import Evaluator
from heca.learning.buffers.stream_buffer import StreamBuffer
from heca.learning.buffers.fair_buffer import FairBuffer
from heca.learning.ppo import PPO
from heca.scenes.ogbench.scene import OGBenchScene

agents = [
    TapasAgent.Config(
        tag="open_drawer",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
    TapasAgent.Config(
        tag="close_drawer",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
    TapasAgent.Config(
        tag="open_window",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
    TapasAgent.Config(
        tag="close_window",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
    TapasAgent.Config(
        tag="lock_left_button",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
    TapasAgent.Config(
        tag="lock_right_button",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
    TapasAgent.Config(
        tag="unlock_left_button",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
    TapasAgent.Config(
        tag="unlock_right_button",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
    TapasAgent.Config(
        tag="move_block",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
    # TapasAgent.Config(
    #    tag="move_ee",
    #    scene=OGBenchScene.Config(),
    #    use_gt=True,
    # ),
]


heca_cfg = Heca.Config(
    tag="test",
    agents=agents,
    learner=PPO.Config(
        tag="test",
        # buffer=APPOBuffer.Config(),
        buffer=FairBuffer.Config(),
        normalize_rewards=False,
    ),
)

heca = Heca.get(heca_cfg)
heca.train(OGBenchScene.Config())
