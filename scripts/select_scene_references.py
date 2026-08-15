from heca.scenes.ogbench.sceneog import OGSceneOG
from heca.guis.scene_sample_selector import SceneRefSelector

selector_cfg = SceneRefSelector.Config(
    scene=OGSceneOG.Config(),
)
selector = SceneRefSelector.get(selector_cfg)
selector.run()
