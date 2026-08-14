from heca.agents.experts.tapas import TapasAgent
from heca.scenes.ogbench.scene import OGScene
from heca.guis.demo_selector import TapasDemoSelector

cfg = TapasDemoSelector.Config(
    agent=TapasAgent.Config(
        tag="close_drawer",
        scene=OGScene.Config(),
    ),
)
selector = TapasDemoSelector.get(cfg)

selector.run()
