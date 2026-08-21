from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene import OGScene
from heca.guis.tapas_agent_tester import TapasManualExecuter

cfg = TapasManualExecuter.Config(
    agents=[
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
        TapasExpert.Config(
            tag="move_ee",
            scene=OGScene.Config(),
        ),
    ],
    scene=OGScene.Config(),
)
tester = TapasManualExecuter.get(cfg)
tester.run()
