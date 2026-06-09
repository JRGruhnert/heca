from heca.agents.experts.tapas import TapasAgent
from heca.environment.scenes.ogbench.scene import OGBenchScene
from heca.guis.demo_selector import DemoSelector

cfg = DemoSelector.Config(
    agent=TapasAgent.Config(
        folder="press_right_button",
        scene=OGBenchScene.Config(),
    ),
    dataset_name="visual-scene-play-256-v0.h5",
)
selector = DemoSelector.create(cfg)

selector.run()
