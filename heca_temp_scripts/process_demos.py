from heca.agents.experts.tapas import TapasAgent
from heca.environment.scenes.ogbench.scene import OGBenchScene
from heca.guis.tapas_demo_processor import TapasDemoProcessor

cfg = TapasDemoProcessor.Config(
    agent=TapasAgent.Config(
        folder="move_block",
        scene=OGBenchScene.Config(),
    )
)
selector = TapasDemoProcessor.get(cfg)

selector.run()
