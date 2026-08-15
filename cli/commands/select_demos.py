from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene import OGScene
from heca.guis.demo_selector import TapasDemoSelector

cfg = TapasDemoSelector.Config(
    agent=TapasExpert.Config(
        tag="close_drawer",
        scene=OGScene.Config(),
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
