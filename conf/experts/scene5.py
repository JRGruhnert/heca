from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene5 import OGScene5

button0_s0_s1 = TapasExpert.Config(
    label="scene5",
    tag="button0_s0_s1",
    scene=OGScene5.Config(),
    gt_frames=[["ee_init", "button1", "button0"], ["button1", "button0", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)
button0_s1_s0 = TapasExpert.Config(
    label="scene5",
    tag="button0_s1_s0",
    scene=OGScene5.Config(),
    gt_frames=[["ee_init", "button1", "button0"], ["button1", "button0", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)

button1_s2_s0 = TapasExpert.Config(
    label="scene5",
    tag="button1_s2_s0",
    scene=OGScene5.Config(),
    gt_frames=[["ee_init", "button1"], ["button1", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)

button1_s0_s1 = TapasExpert.Config(
    label="scene5",
    tag="button1_s0_s1",
    scene=OGScene5.Config(),
    gt_frames=[["ee_init", "button1"], ["button1", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)

button1_s1_s2 = TapasExpert.Config(
    label="scene5",
    tag="button1_s1_s2",
    scene=OGScene5.Config(),
    gt_frames=[["ee_init", "button1"], ["button1", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)

button2_s1_s0 = TapasExpert.Config(
    label="scene5",
    tag="button2_s1_s0",
    scene=OGScene5.Config(),
    gt_frames=[["ee_init", "button2", "button0"], ["button2", "button0", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)

button2_s0_s1 = TapasExpert.Config(
    label="scene5",
    tag="button2_s0_s1",
    scene=OGScene5.Config(),
    gt_frames=[["ee_init", "button2", "button0"], ["button2", "button0", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5],
)

slider0_a_b = TapasExpert.Config(
    label="scene5",
    tag="slider0_a_b",
    scene=OGScene5.Config(),
    gt_frames=[
        ["ee_init", "slider0"],
        ["slider0"],
        ["slider0", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)

faucet0_b_a = TapasExpert.Config(
    label="scene5",
    tag="faucet0_b_a",
    scene=OGScene5.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0"],
        ["faucet0", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)


slider0_b_a = TapasExpert.Config(
    label="scene5",
    tag="slider0_b_a",
    scene=OGScene5.Config(),
    gt_frames=[
        ["ee_init", "slider0"],
        ["slider0"],
        ["slider0", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)


faucet0_a_b = TapasExpert.Config(
    label="scene5",
    tag="faucet0_a_b",
    scene=OGScene5.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0"],
        ["faucet0", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)

drawer0_a_b = TapasExpert.Config(
    label="scene5",
    tag="drawer0_a_b",
    scene=OGScene5.Config(),
    gt_frames=[
        ["ee_init", "drawer0"],
        ["drawer0", "button0", "button1"],
        ["drawer0", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)

drawer0_b_a = TapasExpert.Config(
    label="scene5",
    tag="drawer0_b_a",
    scene=OGScene5.Config(),
    gt_frames=[
        ["ee_init", "drawer0"],
        ["drawer0", "button0", "button1"],
        ["drawer0", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)


agents = [
    slider0_a_b,
    faucet0_b_a,
    button0_s0_s1,
    button2_s1_s0,
    button2_s0_s1,
    slider0_b_a,
    button1_s0_s1,
    faucet0_a_b,
    drawer0_a_b,
    drawer0_b_a,
    button1_s2_s0,
    button1_s1_s2,
    button0_s1_s0,
]
