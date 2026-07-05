from heca.agents.experts.tapas import TapasAgent
from heca.scenes.ogbench.scene import OGBenchScene
from heca.guis.tapas_demo_processor import TapasDemoProcessor

cfg = TapasDemoProcessor.Config(
    agent=TapasAgent.Config(
        tag="move_block_drawer",
        scene=OGBenchScene.Config(),
    )
)
selector = TapasDemoProcessor.get(cfg)

selector.run()
