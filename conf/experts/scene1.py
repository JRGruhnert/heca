from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene1 import OGScene1

button0_s2_s0 = TapasExpert.Config(
    label="scene1",
    tag="button0_s2_s0",
    scene=OGScene1.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button0_s0_s1 = TapasExpert.Config(
    label="scene1",
    tag="button0_s0_s1",
    scene=OGScene1.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button0_s1_s2 = TapasExpert.Config(
    label="scene1",
    tag="button0_s1_s2",
    scene=OGScene1.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

faucet0_a_b = TapasExpert.Config(
    label="scene1",
    tag="faucet0_a_b",
    scene=OGScene1.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0", "button0"],
        ["faucet0", "button0"],
        ["faucet0", "button0"],
        ["faucet0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)
faucet0_b_a = TapasExpert.Config(
    label="scene1",
    tag="faucet0_b_a",
    scene=OGScene1.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0", "button0"],
        ["faucet0", "button0"],
        ["faucet0", "button0"],
        ["faucet0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)


faucet1_b_a = TapasExpert.Config(
    label="scene1",
    tag="faucet1_b_a",
    scene=OGScene1.Config(),
    gt_frames=[
        ["ee_init", "faucet1"],
        ["faucet1", "faucet0", "button0"],
        ["faucet1", "faucet0", "button0"],
        ["faucet1", "faucet0", "button0"],
        ["faucet1", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20],
    snap_ee_actions=False,
)

faucet1_a_b = TapasExpert.Config(
    label="scene1",
    tag="faucet1_a_b",
    scene=OGScene1.Config(),
    gt_frames=[
        ["ee_init", "faucet1"],
        ["faucet1", "faucet0", "button0"],
        ["faucet1", "faucet0", "button0"],
        ["faucet1", "faucet0", "button0"],
        ["faucet1", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    snap_ee_actions=False,
)


cube0_base_base = TapasExpert.Config(
    label="scene1",
    tag="cube0_base_base",
    scene=OGScene1.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0"],
        ["cube0"],
        ["cube0", "cube0_target"],
        ["cube0_target"],
        ["cube0_target"],
        ["cube0_target", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20],
    fix_bimodal=True,
)

cube0_base_shelf0 = TapasExpert.Config(
    label="scene1",
    tag="cube0_base_shelf0",
    scene=OGScene1.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0"],
        ["cube0"],
        ["cube0", "shelf0"],
        ["shelf0"],
        ["shelf0"],
        ["shelf0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    fix_bimodal=True,
)

agents = [
    faucet0_b_a,
    faucet1_b_a,
    cube0_base_base,
    button0_s2_s0,
    button0_s0_s1,
    button0_s1_s2,
    faucet1_a_b,
    cube0_base_shelf0,
    faucet0_a_b,
]
