from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene8 import OGScene8

slider0_a_b = TapasExpert.Config(
    label="scene8",
    tag="slider0_a_b",
    scene=OGScene8.Config(),
    gt_frames=[
        ["ee_init", "slider0"],
        ["slider0", "button0"],
        ["slider0", "button0"],
        ["slider0", "button0"],
        ["slider0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    fix_bimodal=True,
)

slider0_b_a = TapasExpert.Config(
    label="scene8",
    tag="slider0_b_a",
    scene=OGScene8.Config(),
    gt_frames=[
        ["ee_init", "slider0"],
        ["slider0", "button0"],
        ["slider0", "button0"],
        ["slider0", "button0"],
        ["slider0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20],
    fix_bimodal=True,
)
slider1_a_b = TapasExpert.Config(
    label="scene8",
    tag="slider1_a_b",
    scene=OGScene8.Config(),
    gt_frames=[
        ["ee_init", "slider1"],
        ["slider1", "button0"],
        ["slider1", "button0"],
        ["slider1", "button0"],
        ["slider1", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21],
    fix_bimodal=True,
)

slider1_b_a = TapasExpert.Config(
    label="scene8",
    tag="slider1_b_a",
    scene=OGScene8.Config(),
    gt_frames=[
        ["ee_init", "slider1"],
        ["slider1", "button0"],
        ["slider1", "button0"],
        ["slider1", "button0"],
        ["slider1", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    fix_bimodal=True,
)

drawer0_a_b = TapasExpert.Config(
    label="scene8",
    tag="drawer0_a_b",
    scene=OGScene8.Config(),
    gt_frames=[
        ["ee_init", "drawer0"],
        ["drawer0", "slider1", "slider0"],
        ["drawer0", "slider1", "slider0"],
        ["drawer0", "slider1", "slider0"],
        ["drawer0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    fix_bimodal=True,
)
drawer0_b_a = TapasExpert.Config(
    label="scene8",
    tag="drawer0_b_a",
    scene=OGScene8.Config(),
    gt_frames=[
        ["ee_init", "drawer0"],
        ["drawer0", "slider1", "slider0"],
        ["drawer0", "slider1", "slider0"],
        ["drawer0", "slider1", "slider0"],
        ["drawer0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    fix_bimodal=True,
)

cube0_base_base = TapasExpert.Config(
    label="scene8",
    tag="cube0_base_base",
    scene=OGScene8.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0"],
        ["cube0"],
        ["cube0", "cube0_target"],
        ["cube0_target"],
        ["cube0_target"],
        ["cube0_target", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22],
    fix_bimodal=True,
)

cube0_base_drawer0 = TapasExpert.Config(
    label="scene8",
    tag="cube0_base_drawer0",
    scene=OGScene8.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0"],
        ["cube0"],
        ["cube0", "drawer0"],
        ["drawer0"],
        ["drawer0"],
        ["drawer0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 22],
    fix_bimodal=True,
)


button0_s2_s0 = TapasExpert.Config(
    label="scene8",
    tag="button0_s2_s0",
    scene=OGScene8.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button0_s0_s1 = TapasExpert.Config(
    label="scene8",
    tag="button0_s0_s1",
    scene=OGScene8.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button0_s1_s2 = TapasExpert.Config(
    label="scene8",
    tag="button0_s1_s2",
    scene=OGScene8.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20],
    snap_ee_actions=False,
)
agents = [
    slider0_b_a,
    slider0_a_b,
    slider1_b_a,
    slider1_a_b,
    drawer0_a_b,
    drawer0_b_a,
    cube0_base_base,
    cube0_base_drawer0,
    button0_s2_s0,
    button0_s0_s1,
    button0_s1_s2,
]
