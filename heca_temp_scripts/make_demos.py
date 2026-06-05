from heca.guis.demo_selector import DemoSelector

cfg = DemoSelector.Config(
    dataset_name="visual-scene-play-256-v0.h5",
    demos_name="press_left_button",
)
selector = DemoSelector.create(cfg)

selector.run()
