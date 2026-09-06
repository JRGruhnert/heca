from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene6 import OGScene6

button0_s2_s0 = TapasExpert.Config(
    label="scene6",
    tag="button0_s2_s0",
    scene=OGScene6.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button0_s0_s1 = TapasExpert.Config(
    label="scene6",
    tag="button0_s0_s1",
    scene=OGScene6.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button0_s1_s2 = TapasExpert.Config(
    label="scene6",
    tag="button0_s1_s2",
    scene=OGScene6.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 17, 18, 19, 20, 21, 22],
    snap_ee_actions=False,
)


button1_s1_s0 = TapasExpert.Config(
    label="scene6",
    tag="button1_s1_s0",
    scene=OGScene6.Config(),
    gt_frames=[["ee_init", "button1", "button0"], ["button1", "button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button1_s0_s1 = TapasExpert.Config(
    label="scene6",
    tag="button1_s0_s1",
    scene=OGScene6.Config(),
    gt_frames=[["ee_init", "button1", "button0"], ["button1", "button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

faucet0_a_b = TapasExpert.Config(
    label="scene6",
    tag="faucet0_a_b",
    scene=OGScene6.Config(),
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
    label="scene6",
    tag="faucet0_b_a",
    scene=OGScene6.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0", "button0", "button1"],
        ["faucet0", "button0", "button1"],
        ["faucet0", "button0", "button1"],
        ["faucet0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    snap_ee_actions=False,
)

lid0_base_box0 = TapasExpert.Config(
    label="scene6",
    tag="lid0_base_box0",
    scene=OGScene6.Config(),
    gt_frames=[
        ["ee_init", "lid0"],
        ["lid0"],
        ["lid0"],
        ["lid0", "box0"],
        ["box0"],
        ["box0"],
        ["box0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    fix_bimodal=True,
    pos_only=False,
)

lid0_base_base = TapasExpert.Config(
    label="scene6",
    tag="lid0_base_base",
    scene=OGScene6.Config(),
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
    pos_only=False,
)

slider0_a_b = TapasExpert.Config(
    label="scene6",
    tag="slider0_a_b",
    scene=OGScene6.Config(),
    gt_frames=[
        ["ee_init", "slider0"],
        ["slider0", "button0", "button1"],
        ["slider0", "button0", "button1"],
        ["slider0", "button0", "button1"],
        ["slider0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    fix_bimodal=True,
)

slider0_b_a = TapasExpert.Config(
    label="scene6",
    tag="slider0_b_a",
    scene=OGScene6.Config(),
    gt_frames=[
        ["ee_init", "slider0"],
        ["slider0", "button0", "button1"],
        ["slider0", "button0", "button1"],
        ["slider0", "button0", "button1"],
        ["slider0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    fix_bimodal=True,
)

agents = [
    button0_s2_s0,
    button0_s0_s1,
    button0_s1_s2,
    button1_s0_s1,
    button1_s1_s0,
    slider0_a_b,
    slider0_b_a,
    faucet0_a_b,
    faucet0_b_a,
    lid0_base_box0,
    lid0_base_base,
]
