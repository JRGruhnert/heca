from heca.agents.experts.tapas import TapasAgent
from heca.environment.scenes.ogbench.scene import OGBenchScene
from heca.guis.agent_tester import AgentTester

cfg = AgentTester.Config(
    agents=[
        TapasAgent.Config(
            folder="open_drawer",
            scene=OGBenchScene.Config(),
        ),
        TapasAgent.Config(
            folder="close_drawer",
            scene=OGBenchScene.Config(),
        ),
    ],
    scene=OGBenchScene.Config(),
)
tester = AgentTester.get(cfg)
tester.run()
