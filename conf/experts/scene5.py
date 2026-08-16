from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene5 import OGScene5

slider0_a_b = TapasExpert.Config(
    tag="slider0_a_b",
    scene=OGScene5.Config(),
    gt_frames=[
        ["ee_init", "slider0"],
        ["slider0"],
        ["slider0", "ee_target"],
    ],
)

faucet0_b_a = TapasExpert.Config(
    tag="faucet0_b_a",
    scene=OGScene5.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0"],
        ["faucet0", "ee_target"],
    ],
)

button0_s0_s1 = TapasExpert.Config(
    tag="button0_s0_s1",
    scene=OGScene5.Config(),
    gt_frames=[["ee_init", "button1", "button0"], ["button1", "button0", "ee_target"]],
)

button2_s1_s0 = TapasExpert.Config(
    tag="button2_s1_s0",
    scene=OGScene5.Config(),
    gt_frames=[["ee_init", "button2", "button0"], ["button2", "button0", "ee_target"]],
)

button2_s0_s1 = TapasExpert.Config(
    tag="button2_s0_s1",
    scene=OGScene5.Config(),
    gt_frames=[["ee_init", "button2", "button0"], ["button2", "button0", "ee_target"]],
)

slider0_b_a = TapasExpert.Config(
    tag="slider0_b_a",
    scene=OGScene5.Config(),
    gt_frames=[
        ["ee_init", "slider0"],
        ["slider0"],
        ["slider0", "ee_target"],
    ],
)

button1_s0_s1 = TapasExpert.Config(
    tag="button1_s0_s1",
    scene=OGScene5.Config(),
    gt_frames=[["ee_init", "button1"], ["button1", "ee_target"]],
)

faucet0_a_b = TapasExpert.Config(
    tag="faucet0_a_b",
    scene=OGScene5.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0"],
        ["faucet0", "ee_target"],
    ],
)

drawer0_a_b = TapasExpert.Config(
    tag="drawer0_a_b",
    scene=OGScene5.Config(),
    gt_frames=[
        ["ee_init", "drawer0"],
        ["drawer0", "button0"],
        ["drawer0", "ee_target"],
    ],
)

drawer0_b_a = TapasExpert.Config(
    tag="drawer0_b_a",
    scene=OGScene5.Config(),
    gt_frames=[
        ["ee_init", "drawer0"],
        ["drawer0", "button0"],
        ["drawer0", "ee_target"],
    ],
)

button1_s2_s0 = TapasExpert.Config(
    tag="button1_s2_s0",
    scene=OGScene5.Config(),
    gt_frames=[["ee_init", "button1"], ["button1", "ee_target"]],
)

button1_s1_s2 = TapasExpert.Config(
    tag="button1_s1_s2",
    scene=OGScene5.Config(),
    gt_frames=[["ee_init", "button1"], ["button1", "ee_target"]],
)

button0_s1_s0 = TapasExpert.Config(
    tag="button0_s1_s0",
    scene=OGScene5.Config(),
    gt_frames=[["ee_init", "button1", "button0"], ["button1", "button0", "ee_target"]],
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
