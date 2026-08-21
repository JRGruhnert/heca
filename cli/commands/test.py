from heca.agents.agent import Agent
from heca.experts.tapas import TapasExpert

# from heca.agents.heca import Heca
from heca.agents.heca_old import Heca
from heca.conditions.evaluator import Evaluator
from heca.learning.appo import APPO
from heca.learning.buffers.stream_buffer import StreamBuffer
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
    learner=PPO.Config(tag="test"),
)
heca2_cfg = Heca.Config(
    tag="test2",
    agents=agents,
    learner=APPO.Config(tag="test"),
)

# heca = Agent.get(heca_cfg, auto_load=False)
# heca.conditions

agent = Agent.get(
    TapasExpert.Config(
        tag="unlock_right_button",
        scene=OGScene.Config(),
    ),
)

print(agent.conditions[0].elabels)
parameters = agent.conditions[0].pre.models["button_1"].get_parameters()
print(parameters)
print(parameters["measurement"])
