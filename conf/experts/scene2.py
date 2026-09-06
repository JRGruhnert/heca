from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene2 import OGScene2

button0_s1_s2 = TapasExpert.Config(
    label="scene2",
    tag="button0_s1_s2",
    scene=OGScene2.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)
button0_s2_s0 = TapasExpert.Config(
    label="scene2",
    tag="button0_s2_s0",
    scene=OGScene2.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button0_s0_s1 = TapasExpert.Config(
    label="scene2",
    tag="button0_s0_s1",
    scene=OGScene2.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button1_s0_s1 = TapasExpert.Config(
    label="scene2",
    tag="button1_s0_s1",
    scene=OGScene2.Config(),
    gt_frames=[["ee_init", "button1", "button0"], ["button1", "button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button1_s1_s0 = TapasExpert.Config(
    label="scene2",
    tag="button1_s1_s0",
    scene=OGScene2.Config(),
    gt_frames=[["ee_init", "button1", "button0"], ["button1", "button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)


peg0_base_base = TapasExpert.Config(
    label="scene2",
    tag="peg0_base_base",
    scene=OGScene2.Config(),
    gt_frames=[
        ["ee_init", "peg0"],
        ["peg0"],
        ["peg0"],
        ["peg0", "peg0_target"],
        ["peg0_target"],
        ["peg0_target"],
        ["peg0_target", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20],
    fix_bimodal=True,
)

lid0_base_base = TapasExpert.Config(
    label="scene2",
    tag="lid0_base_base",
    scene=OGScene2.Config(),
    gt_frames=[
        ["ee_init", "lid0"],
        ["lid0"],
        ["lid0"],
        ["lid0", "lid0_target"],
        ["lid0_target"],
        ["lid0_target"],
        ["lid0_target", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    fix_bimodal=True,
)

faucet0_a_b = TapasExpert.Config(
    label="scene2",
    tag="faucet0_a_b",
    scene=OGScene2.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0", "button0", "button1"],
        ["faucet0", "button0", "button1"],
        ["faucet0", "button0", "button1"],
        ["faucet0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

faucet0_b_a = TapasExpert.Config(
    label="scene2",
    tag="faucet0_b_a",
    scene=OGScene2.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0", "button0", "button1"],
        ["faucet0", "button0", "button1"],
        ["faucet0", "button0", "button1"],
        ["faucet0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    snap_ee_actions=False,
)

lid0_base_box0 = TapasExpert.Config(
    label="scene2",
    tag="lid0_base_box0",
    scene=OGScene2.Config(),
    gt_frames=[
        ["ee_init", "lid0"],
        ["lid0"],
        ["lid0"],
        ["lid0", "box0"],
        ["box0"],
        ["box0"],
        ["box0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21],
    fix_bimodal=True,
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
