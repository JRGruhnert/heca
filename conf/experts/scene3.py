from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene3 import OGScene3

button2_s1_s0 = TapasExpert.Config(
    tag="button2_s1_s0",
    scene=OGScene3.Config(),
    gt_frames=[["ee_init", "button2", "button0"], ["button2", "button0", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)

button1_s0_s1 = TapasExpert.Config(
    tag="button1_s0_s1",
    scene=OGScene3.Config(),
    gt_frames=[["ee_init", "button1"], ["button1", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)

button2_s0_s1 = TapasExpert.Config(
    tag="button2_s0_s1",
    scene=OGScene3.Config(),
    gt_frames=[["ee_init", "button2", "button0"], ["button2", "button0", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)

cube0_base_base = TapasExpert.Config(
    tag="cube0_base_base",
    scene=OGScene3.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0", "cube0_target"],
        ["cube0_target", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)

cube0_base_shelf0 = TapasExpert.Config(
    tag="cube0_base_shelf0",
    scene=OGScene3.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0", "shelf0"],
        ["shelf0", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)

button1_s1_s0 = TapasExpert.Config(
    tag="button1_s1_s0",
    scene=OGScene3.Config(),
    gt_frames=[["ee_init", "button1"], ["button1", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)

drawer0_b_a = TapasExpert.Config(
    tag="drawer0_b_a",
    scene=OGScene3.Config(),
    gt_frames=[
        ["ee_init", "drawer0"],
        ["drawer0", "button0"],
        ["drawer0", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)

button0_s1_s0 = TapasExpert.Config(
    tag="button0_s1_s0",
    scene=OGScene3.Config(),
    gt_frames=[["ee_init", "button1", "button0"], ["button1", "button0", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)
button0_s0_s1 = TapasExpert.Config(
    tag="button0_s0_s1",
    scene=OGScene3.Config(),
    gt_frames=[["ee_init", "button1", "button0"], ["button1", "button0", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)

drawer0_a_b = TapasExpert.Config(
    tag="drawer0_a_b",
    scene=OGScene3.Config(),
    gt_frames=[
        ["ee_init", "drawer0"],
        ["drawer0", "button0"],
        ["drawer0", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)

cube0_base_drawer0 = TapasExpert.Config(
    tag="cube0_base_drawer0",
    scene=OGScene3.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0", "drawer0"],
        ["drawer0", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)


agents = [
    button0_s1_s0,
    button0_s0_s1,
    button2_s1_s0,
    button2_s0_s1,
    button1_s0_s1,
    button1_s1_s0,
    cube0_base_base,
    cube0_base_shelf0,
    drawer0_b_a,
    drawer0_a_b,
    cube0_base_drawer0,
]
