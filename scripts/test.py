from heca.agents.agent import Agent
from heca.agents.experts.tapas import TapasAgent

# from heca.agents.heca import Heca
from heca.agents.heca import Heca
from heca.conditions.evaluator import Evaluator
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
)

# heca = Agent.get(heca_cfg, auto_load=False)
# heca.conditions

agent = Agent.get(
    TapasAgent.Config(
        tag="unlock_right_button",
        scene=OGBenchScene.Config(),
        use_gt=True,
    ),
)

print(agent.conditions[0].elabels)
parameters = agent.conditions[0].pre.model["button_1"].get_parameters()
print(parameters)
print(parameters["measurement"])
