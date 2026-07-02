from heca.agents.experts.tapas import TapasAgent
from heca.scenes.ogbench.scene import OGBenchScene

close_drawer = TapasAgent.Config(
    folder="close_drawer",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 1], [1, 3], [1, 7]],
)

open_drawer = TapasAgent.Config(
    folder="open_drawer",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 1], [1, 3], [1, 7]],
)

close_window = TapasAgent.Config(
    folder="close_window",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 2], [2, 4], [2, 7]],
)

open_window = TapasAgent.Config(
    folder="open_window",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 2], [2, 4], [2, 7]],
)

lock_left = TapasAgent.Config(
    folder="lock_left_button",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 3], [3, 7]],
)

unlock_left = TapasAgent.Config(
    folder="unlock_left_button",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 3], [3, 7]],
)

lock_right = TapasAgent.Config(
    folder="lock_right_button",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 4], [4, 7]],
)

unlock_right = TapasAgent.Config(
    folder="unlock_right_button",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 4], [4, 7]],
)


move_block_drawer = TapasAgent.Config(
    folder="move_block_drawer",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 5], [3, 5], [3, 7]],
)

move_block = TapasAgent.Config(
    folder="move_block",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 5], [5, 6], [6, 7]],
)

move_ee = TapasAgent.Config(
    folder="move_ee",
    scene=OGBenchScene.Config(),
    use_gt=True,
    gt_frames=[[0, 7]],
)
