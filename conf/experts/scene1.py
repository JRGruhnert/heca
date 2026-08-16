from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene1 import OGScene1

faucet0_b_a = TapasExpert.Config(
    tag="faucet0_b_a",
    scene=OGScene1.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0", "shelf0"],
        ["faucet0", "ee_target"],
    ],
)

faucet1_b_a = TapasExpert.Config(
    tag="faucet1_b_a",
    scene=OGScene1.Config(),
    gt_frames=[
        ["ee_init", "faucet1"],
        ["faucet1", "faucet0"],
        ["faucet1", "ee_target"],
    ],
)

faucet1_a_b = TapasExpert.Config(
    tag="faucet1_a_b",
    scene=OGScene1.Config(),
    gt_frames=[
        ["ee_init", "faucet1"],
        ["faucet1", "faucet0"],
        ["faucet1", "ee_target"],
    ],
)

faucet0_a_b = TapasExpert.Config(
    tag="faucet0_a_b",
    scene=OGScene1.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0", "shelf0"],
        ["faucet0", "ee_target"],
    ],
)

cube0_base_base = TapasExpert.Config(
    tag="cube0_base_base",
    scene=OGScene1.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0", "cube0_target"],
        ["cube0_target", "ee_target"],
    ],
)

button0_s1_s0 = TapasExpert.Config(
    tag="button0_s1_s0",
    scene=OGScene1.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
)

button0_s0_s1 = TapasExpert.Config(
    tag="button0_s0_s1",
    scene=OGScene1.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
)

cube0_base_shelf0 = TapasExpert.Config(
    tag="cube0_base_shelf0",
    scene=OGScene1.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0", "shelf0"],
        ["shelf0", "ee_target"],
    ],
)


agents = [
    faucet0_b_a,
    faucet1_b_a,
    cube0_base_base,
    button0_s1_s0,
    button0_s0_s1,
    faucet1_a_b,
    cube0_base_shelf0,
    faucet0_a_b,
]
