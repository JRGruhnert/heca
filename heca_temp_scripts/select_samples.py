from heca.scenes.calvin.scene import CalvinScene
from heca.scenes.ogbench.scene import OGBenchScene
from heca.guis.scene_sample_selector import SceneSampleSelector

selector_cfg = SceneSampleSelector.Config(
    scene=OGBenchScene.Config(),
)
selector = SceneSampleSelector.get(selector_cfg)
selector.run()
