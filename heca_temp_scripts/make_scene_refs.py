from heca.environment.scenes.calvin.scene import CalvinScene
from heca.guis.image_selector import ImageSelector

selector_cfg = ImageSelector.Config(
    scene=CalvinScene.Config(),
)
selector = ImageSelector.create(selector_cfg)
selector.run()
