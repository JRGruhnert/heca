from heca.agents.experts.tapas import TapasAgent
from heca.scenes.ogbench.scene import OGBenchScene
from heca.guis.tapas_demo_selector import TapasDemoSelector

cfg = TapasDemoSelector.Config(
    agent=TapasAgent.Config(
        folder="close_drawer",
        scene=OGBenchScene.Config(),
    ),
)
selector = TapasDemoSelector.get(cfg)

selector.run()

# TODO:


# lock_right_button
# unlock_right_button
# move_block_drawer <- need to rename
# lock_left_button
# unlock_left_button
# open_window
# close_window
# move_block
# move_ee
# open_drawer
# close_drawer
