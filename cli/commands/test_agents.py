from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene import OGScene
from heca.guis.tapas_agent_tester import AgentTester

cfg = AgentTester.Config(
    agents=[
        TapasExpert.Config(
            tag="open_drawer",
            scene=OGScene.Config(),
            use_gt=True,
        ),
        TapasExpert.Config(
            tag="close_drawer",
            scene=OGScene.Config(),
            use_gt=True,
        ),
        TapasExpert.Config(
            tag="open_window",
            scene=OGScene.Config(),
            use_gt=True,
        ),
        TapasExpert.Config(
            tag="close_window",
            scene=OGScene.Config(),
            use_gt=True,
        ),
        TapasExpert.Config(
            tag="lock_left_button",
            scene=OGScene.Config(),
            use_gt=True,
        ),
        TapasExpert.Config(
            tag="lock_right_button",
            scene=OGScene.Config(),
            use_gt=True,
        ),
        TapasExpert.Config(
            tag="unlock_left_button",
            scene=OGScene.Config(),
            use_gt=True,
        ),
        TapasExpert.Config(
            tag="unlock_right_button",
            scene=OGScene.Config(),
            use_gt=True,
        ),
        TapasExpert.Config(
            tag="move_block",
            scene=OGScene.Config(),
            use_gt=True,
        ),
        TapasExpert.Config(
            tag="move_ee",
            scene=OGScene.Config(),
            use_gt=True,
        ),
    ],
    scene=OGScene.Config(),
)
tester = AgentTester.get(cfg)
tester.run()
