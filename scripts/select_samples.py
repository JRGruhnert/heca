from heca.scenes.calvin.scene import CalvinScene
from heca.scenes.ogbench.scene import OGScene
from heca.guis.scene_sample_selector import SceneSampleSelector

selector_cfg = SceneSampleSelector.Config(
    scene=OGScene.Config(),
)
selector = SceneSampleSelector.get(selector_cfg)
selector.run()
