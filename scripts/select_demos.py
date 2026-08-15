from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.sceneog import OGSceneOG
from heca.guis.demo_selector import TapasDemoSelector

cfg = TapasDemoSelector.Config(
    agent=TapasExpert.Config(
        tag="close_drawer",
        scene=OGSceneOG.Config(),
    ),
)
selector = TapasDemoSelector.get(cfg)

selector.run()
