from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene2 import OGScene2

faucet0_b_a = TapasExpert.Config(
    tag="faucet0_b_a",
    scene=OGScene2.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0", "button0", "button1"],
        ["faucet0", "ee_target"],
    ],
)

button0_s1_s2 = TapasExpert.Config(
    tag="button0_s1_s2",
    scene=OGScene2.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
)

peg0_base_base = TapasExpert.Config(
    tag="peg0_base_base",
    scene=OGScene2.Config(),
    gt_frames=[
        ["ee_init", "peg0"],
        ["peg0", "peg0_target"],
        ["peg0_target", "ee_target"],
    ],
)

button1_s1_s0 = TapasExpert.Config(
    tag="button1_s1_s0",
    scene=OGScene2.Config(),
    gt_frames=[["ee_init", "button1"], ["button1", "ee_target"]],
)

lid0_base_base = TapasExpert.Config(
    tag="lid0_base_base",
    scene=OGScene2.Config(),
    gt_frames=[
        ["ee_init", "lid0"],
        ["lid0", "lid0_target"],
        ["lid0_target", "ee_target"],
    ],
)

faucet0_a_b = TapasExpert.Config(
    tag="faucet0_a_b",
    scene=OGScene2.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0", "button0", "button1"],
        ["faucet0", "ee_target"],
    ],
)

lid0_base_box0 = TapasExpert.Config(
    tag="lid0_base_box0",
    scene=OGScene2.Config(),
    gt_frames=[
        ["ee_init", "lid0"],
        ["lid0", "box0"],
        ["box0", "ee_target"],
    ],
)

button0_s0_s1 = TapasExpert.Config(
    tag="button0_s0_s1",
    scene=OGScene2.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
)

button1_s0_s1 = TapasExpert.Config(
    tag="button1_s0_s1",
    scene=OGScene2.Config(),
    gt_frames=[["ee_init", "button1"], ["button1", "ee_target"]],
)

button0_s2_s0 = TapasExpert.Config(
    tag="button0_s2_s0",
    scene=OGScene2.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
)


agents = [
    faucet0_b_a,
    button0_s1_s2,
    peg0_base_base,
    button1_s1_s0,
    lid0_base_base,
    faucet0_a_b,
    lid0_base_box0,
    button0_s0_s1,
    button1_s0_s1,
    button0_s2_s0,
]
