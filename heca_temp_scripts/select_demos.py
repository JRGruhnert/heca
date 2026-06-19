from heca.agents.experts.tapas import TapasAgent
from heca.environment.scenes.ogbench.scene import OGBenchScene
from heca.guis.tapas_demo_selector import TapasDemoSelector

cfg = TapasDemoSelector.Config(
    agent=TapasAgent.Config(
        folder="move_block_drawer",
        scene=OGBenchScene.Config(),
    ),
)
selector = TapasDemoSelector.get(cfg)

selector.run()

# TODO:
# move_ee
# open_drawer
# close_drawer
# lock_left_button
# lock_right_button
# unlock_left_button
# unlock_right_button
# open_window
# close_window
# move_block
# move_block_drawer
