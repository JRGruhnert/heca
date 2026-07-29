from heca.agents.experts.tapas import TapasAgent
from heca.scenes.ogbench.scene1 import OGScene1
from heca.guis.tapas_agent_tester import AgentTester

cfg = AgentTester.Config(
    agents=[
        TapasAgent.Config(
            tag="open_drawer",
            scene=OGScene1.Config(),
            use_gt=True,
        ),
        # TapasAgent.Config(
        #     tag="close_drawer",
        #     scene=OGBenchScene.Config(),
        #     use_gt=True,
        # ),
        # TapasAgent.Config(
        #     tag="open_window",
        #     scene=OGBenchScene.Config(),
        #     use_gt=True,
        # ),
        # TapasAgent.Config(
        #     tag="close_window",
        #     scene=OGBenchScene.Config(),
        #     use_gt=True,
        # ),
        # TapasAgent.Config(
        #     tag="lock_left_button",
        #     scene=OGBenchScene.Config(),
        #     use_gt=True,
        # ),
        # TapasAgent.Config(
        #     tag="lock_right_button",
        #     scene=OGBenchScene.Config(),
        #     use_gt=True,
        # ),
        # TapasAgent.Config(
        #     tag="unlock_left_button",
        #     scene=OGBenchScene.Config(),
        #     use_gt=True,
        # ),
        # TapasAgent.Config(
        #     tag="unlock_right_button",
        #     scene=OGBenchScene.Config(),
        #     use_gt=True,
        # ),
        # TapasAgent.Config(
        #     tag="move_block",
        #     scene=OGBenchScene.Config(),
        #     use_gt=True,
        # ),
        # TapasAgent.Config(
        #     tag="move_ee",
        #     scene=OGBenchScene.Config(),
        #     use_gt=True,
        # ),
    ],
    scene=OGScene1.Config(),
)
tester = AgentTester.get(cfg)
tester.run()
