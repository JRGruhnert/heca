from heca.agents.experts.tapas import TapasAgent
from heca.scenes.ogbench.scene import OGBenchScene

close_drawer = TapasAgent.Config(
    tag="close_drawer",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 4], [4, 2], [4, 7]],
)

open_drawer = TapasAgent.Config(
    tag="open_drawer",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 4], [4, 2], [4, 7]],
)

close_window = TapasAgent.Config(
    tag="close_window",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 5], [5, 3], [5, 7]],
)

open_window = TapasAgent.Config(
    tag="open_window",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 5], [5, 3], [5, 7]],
)

lock_left = TapasAgent.Config(
    tag="lock_left_button",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 2], [2, 7]],
)

unlock_left = TapasAgent.Config(
    tag="unlock_left_button",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 2], [2, 7]],
)

lock_right = TapasAgent.Config(
    tag="lock_right_button",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 3], [3, 7]],
)

unlock_right = TapasAgent.Config(
    tag="unlock_right_button",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 3], [3, 7]],
)


move_block_drawer = TapasAgent.Config(
    tag="move_block_drawer",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 1], [4, 1], [4, 7]],
)

move_block = TapasAgent.Config(
    tag="move_block",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 1], [1, 6], [6, 7]],
)

move_ee = TapasAgent.Config(
    tag="move_ee",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 7]],
)
