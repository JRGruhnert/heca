from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene1 import OGScene1

close_drawer = TapasExpert.Config(
    tag="close_drawer",
    scene=OGScene1.Config(),
    use_gt=True,
    gt_frames=[[0, 4], [4, 2], [4, 7]],
)

open_drawer = TapasExpert.Config(
    tag="open_drawer",
    scene=OGScene1.Config(),
    use_gt=True,
    gt_frames=[[0, 4], [4, 2], [4, 7]],
)

close_window = TapasExpert.Config(
    tag="close_window",
    scene=OGScene1.Config(),
    use_gt=True,
    gt_frames=[[0, 5], [5, 3], [5, 7]],
)

open_window = TapasExpert.Config(
    tag="open_window",
    scene=OGScene1.Config(),
    use_gt=True,
    gt_frames=[[0, 5], [5, 3], [5, 7]],
)

lock_left = TapasExpert.Config(
    tag="lock_left_button",
    scene=OGScene1.Config(),
    use_gt=True,
    gt_frames=[[0, 2], [2, 7]],
)

unlock_left = TapasExpert.Config(
    tag="unlock_left_button",
    scene=OGScene1.Config(),
    use_gt=True,
    gt_frames=[[0, 2], [2, 7]],
)

lock_right = TapasExpert.Config(
    tag="lock_right_button",
    scene=OGScene1.Config(),
    use_gt=True,
    gt_frames=[[0, 3], [3, 7]],
)

unlock_right = TapasExpert.Config(
    tag="unlock_right_button",
    scene=OGScene1.Config(),
    use_gt=True,
    gt_frames=[[0, 3], [3, 7]],
)


move_block_drawer = TapasExpert.Config(
    tag="move_block_drawer",
    scene=OGScene1.Config(),
    use_gt=True,
    gt_frames=[[0, 1], [4, 1], [4, 7]],
)

move_block = TapasExpert.Config(
    tag="move_block",
    scene=OGScene1.Config(),
    use_gt=True,
    gt_frames=[[0, 1], [1, 6], [6, 7]],
)

move_ee = TapasExpert.Config(
    tag="move_ee",
    scene=OGScene1.Config(),
    use_gt=True,
    gt_frames=[[0, 7]],
)


agents = [
    close_drawer,
    close_window,
    open_drawer,
    open_window,
    lock_left,
    lock_right,
    unlock_left,
    unlock_right,
    move_block,
    move_block_drawer,
]
