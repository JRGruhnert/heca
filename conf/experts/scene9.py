from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene9 import OGScene9

button0_s0_s1 = TapasExpert.Config(
    label="scene9",
    tag="button0_s0_s1",
    scene=OGScene9.Config(),
    gt_frames=[["ee_init", "button0", "faucet1"], ["button0", "faucet1", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button0_s1_s0 = TapasExpert.Config(
    label="scene9",
    tag="button0_s1_s0",
    scene=OGScene9.Config(),
    gt_frames=[["ee_init", "button0", "faucet1"], ["button0", "faucet1", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)
button2_s0_s1 = TapasExpert.Config(
    label="scene9",
    tag="button2_s0_s1",
    scene=OGScene9.Config(),
    gt_frames=[["ee_init", "button2", "faucet0"], ["button2", "faucet0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button2_s1_s0 = TapasExpert.Config(
    label="scene9",
    tag="button2_s1_s0",
    scene=OGScene9.Config(),
    gt_frames=[["ee_init", "button2", "faucet0"], ["button2", "faucet0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)


button1_s2_s0 = TapasExpert.Config(
    label="scene9",
    tag="button1_s2_s0",
    scene=OGScene9.Config(),
    gt_frames=[["ee_init", "button1", "faucet1"], ["button1", "faucet1", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button1_s0_s1 = TapasExpert.Config(
    label="scene9",
    tag="button1_s0_s1",
    scene=OGScene9.Config(),
    gt_frames=[["ee_init", "button1", "faucet1"], ["button1", "faucet1", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button1_s1_s2 = TapasExpert.Config(
    label="scene9",
    tag="button1_s1_s2",
    scene=OGScene9.Config(),
    gt_frames=[["ee_init", "button1", "faucet1"], ["button1", "faucet1", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)
button3_s2_s0 = TapasExpert.Config(
    label="scene9",
    tag="button3_s2_s0",
    scene=OGScene9.Config(),
    gt_frames=[["ee_init", "button3", "faucet0"], ["button3", "faucet0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    snap_ee_actions=False,
)

button3_s0_s1 = TapasExpert.Config(
    label="scene9",
    tag="button3_s0_s1",
    scene=OGScene9.Config(),
    gt_frames=[["ee_init", "button3", "faucet0"], ["button3", "faucet0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button3_s1_s2 = TapasExpert.Config(
    label="scene9",
    tag="button3_s1_s2",
    scene=OGScene9.Config(),
    gt_frames=[["ee_init", "button3", "faucet0"], ["button3", "faucet0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

faucet0_a_b = TapasExpert.Config(
    label="scene9",
    tag="faucet0_a_b",
    scene=OGScene9.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0"],
        ["faucet0"],
        ["faucet0"],
        ["faucet0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20],
    snap_ee_actions=False,
)

faucet0_b_a = TapasExpert.Config(
    label="scene9",
    tag="faucet0_b_a",
    scene=OGScene9.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0"],
        ["faucet0"],
        ["faucet0"],
        ["faucet0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21],
    snap_ee_actions=False,
)

faucet1_a_b = TapasExpert.Config(
    label="scene9",
    tag="faucet1_a_b",
    scene=OGScene9.Config(),
    gt_frames=[
        ["ee_init", "faucet1"],
        ["faucet1", "button1"],
        ["faucet1", "button1"],
        ["faucet1", "button1"],
        ["faucet1", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

faucet1_b_a = TapasExpert.Config(
    label="scene9",
    tag="faucet1_b_a",
    scene=OGScene9.Config(),
    gt_frames=[
        ["ee_init", "faucet1"],
        ["faucet1", "button1"],
        ["faucet1", "button1"],
        ["faucet1", "button1"],
        ["faucet1", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)
agents = [
    button0_s0_s1,
    button0_s1_s0,
    button1_s0_s1,
    button1_s1_s2,
    button1_s2_s0,
    button2_s0_s1,
    button2_s1_s0,
    button3_s0_s1,
    button3_s1_s2,
    button3_s2_s0,
    faucet0_a_b,
    faucet0_b_a,
    faucet1_a_b,
    faucet1_b_a,
]
