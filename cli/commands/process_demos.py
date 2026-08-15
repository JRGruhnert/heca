from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene import OGScene
from heca.guis.tapas_demo_processor import TapasDemoProcessor

cfg = TapasDemoProcessor.Config(
    agent=TapasExpert.Config(
        tag="move_block_drawer",
        scene=OGScene.Config(),
    )
)
selector = TapasDemoProcessor.get(cfg)

selector.run()
