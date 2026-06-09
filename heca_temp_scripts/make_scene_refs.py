from heca.environment.scenes.calvin.scene import CalvinScene
from heca.environment.scenes.ogbench.scene import OGBenchScene
from heca.guis.image_selector import ImageSelector

selector_cfg = ImageSelector.Config(
    scene=OGBenchScene.Config(),
)
selector = ImageSelector.create(selector_cfg)
selector.run()
