from heca.agents.experts.tapas import TapasAgent
from heca.environment.scenes.ogbench.scene import OGBenchScene
from heca.guis.demo_postprocessor import DemoPostProcessor

cfg = DemoPostProcessor.Config(
    agent=TapasAgent.Config(
        folder="move_block_drawer",
        scene=OGBenchScene.Config(),
    )
)
selector = DemoPostProcessor.get(cfg)

selector.run()
