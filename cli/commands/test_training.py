from heca.agents.agent import Agent
from heca.experts.tapas import TapasExpert

# from heca.agents.heca import Heca
from heca.agents.heca_old import Heca
from heca.conditions.evaluator import Evaluator
from heca.learning.buffers.stream_buffer import StreamBuffer
from heca.learning.buffers.fair_buffer import FairBuffer
from heca.learning.ppo import PPO
from heca.scenes.ogbench.scene import OGScene

agents = [
    TapasExpert.Config(
        tag="open_drawer",
        scene=OGScene.Config(),
    ),
    TapasExpert.Config(
        tag="close_drawer",
        scene=OGScene.Config(),
    ),
    TapasExpert.Config(
        tag="open_window",
        scene=OGScene.Config(),
    ),
    TapasExpert.Config(
        tag="close_window",
        scene=OGScene.Config(),
    ),
    TapasExpert.Config(
        tag="lock_left_button",
        scene=OGScene.Config(),
    ),
    TapasExpert.Config(
        tag="lock_right_button",
        scene=OGScene.Config(),
    ),
    TapasExpert.Config(
        tag="unlock_left_button",
        scene=OGScene.Config(),
    ),
    TapasExpert.Config(
        tag="unlock_right_button",
        scene=OGScene.Config(),
    ),
    TapasExpert.Config(
        tag="move_block",
        scene=OGScene.Config(),
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
    ),
)
