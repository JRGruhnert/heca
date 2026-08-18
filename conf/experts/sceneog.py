from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.sceneog import OGSceneOG

close_drawer = TapasExpert.Config(
    label="sceneog",
    tag="close_drawer",
    scene=OGSceneOG.Config(),
    use_gt=True,
    gt_frames=[
        ["ee_init", "drawer_handle"],
        ["drawer_handle", "button_0"],
        ["drawer_handle", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)

open_drawer = TapasExpert.Config(
    label="sceneog",
    tag="open_drawer",
    scene=OGSceneOG.Config(),
    use_gt=True,
    gt_frames=[
        ["ee_init", "drawer_handle"],
        ["drawer_handle", "button_0"],
        ["drawer_handle", "ee_target"],
    ],
    segment_ids=[0, 1, 4, 6, 7],
)

close_window = TapasExpert.Config(
    label="sceneog",
    tag="close_window",
    scene=OGSceneOG.Config(),
    use_gt=True,
    gt_frames=[
        ["ee_init", "window_handle"],
        ["window_handle", "button_1"],
        ["window_handle", "ee_target"],
    ],
    segment_ids=[0, 4, 5, 6, 7],
)

open_window = TapasExpert.Config(
    label="sceneog",
    tag="open_window",
    scene=OGSceneOG.Config(),
    use_gt=True,
    gt_frames=[
        ["ee_init", "window_handle"],
        ["window_handle", "button_1"],
        ["window_handle", "ee_target"],
    ],
    segment_ids=[3, 4, 5, 10, 13],
)

lock_left = TapasExpert.Config(
    label="sceneog",
    tag="lock_left_button",
    scene=OGSceneOG.Config(),
    use_gt=True,
    gt_frames=[["ee_init", "button_0"], ["button_0", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)

unlock_left = TapasExpert.Config(
    label="sceneog",
    tag="unlock_left_button",
    scene=OGSceneOG.Config(),
    use_gt=True,
    gt_frames=[["ee_init", "button_0"], ["button_0", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)

lock_right = TapasExpert.Config(
    label="sceneog",
    tag="lock_right_button",
    scene=OGSceneOG.Config(),
    use_gt=True,
    gt_frames=[["ee_init", "button_1"], ["button_1", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)

unlock_right = TapasExpert.Config(
    label="sceneog",
    tag="unlock_right_button",
    scene=OGSceneOG.Config(),
    use_gt=True,
    gt_frames=[["ee_init", "button_1"], ["button_1", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)


move_block_drawer = TapasExpert.Config(
    label="sceneog",
    tag="move_block_drawer",
    scene=OGSceneOG.Config(),
    use_gt=True,
    gt_frames=[
        ["ee_init", "block_0"],
        ["drawer_handle", "block_0"],
        ["drawer_handle", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)

move_block = TapasExpert.Config(
    label="sceneog",
    tag="move_block",
    scene=OGSceneOG.Config(),
    use_gt=True,
    gt_frames=[
        ["ee_init", "block_0"],
        ["block_0", "block_0_target"],
        ["block_0_target", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
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
