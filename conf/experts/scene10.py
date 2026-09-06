from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene10 import OGScene10

lid0_base_base = TapasExpert.Config(
    label="scene10",
    tag="lid0_base_base",
    scene=OGScene10.Config(),
    gt_frames=[
        ["ee_init", "lid0"],
        ["lid0"],
        ["lid0"],
        ["lid0", "lid0_target"],
        ["lid0_target"],
        ["lid0_target"],
        ["lid0_target", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20],
    fix_bimodal=True,
)

lid0_base_box0 = TapasExpert.Config(
    label="scene10",
    tag="lid0_base_box0",
    scene=OGScene10.Config(),
    gt_frames=[
        ["ee_init", "lid0"],
        ["lid0"],
        ["lid0"],
        ["lid0", "box0"],
        ["box0"],
        ["box0"],
        ["box0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21],
    fix_bimodal=True,
)

lid0_base_box1 = TapasExpert.Config(
    label="scene10",
    tag="lid0_base_box1",
    scene=OGScene10.Config(),
    gt_frames=[
        ["ee_init", "lid0"],
        ["lid0"],
        ["lid0"],
        ["lid0", "box1"],
        ["box1"],
        ["box1"],
        ["box1", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21],
    fix_bimodal=True,
)


cube0_base_base = TapasExpert.Config(
    label="scene10",
    tag="cube0_base_base",
    scene=OGScene10.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0"],
        ["cube0"],
        ["cube0", "cube0_target"],
        ["cube0_target"],
        ["cube0_target"],
        ["cube0_target", "ee_target"],
    ],
    segment_ids=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 22],
    fix_bimodal=True,
)

cube0_base_box0 = TapasExpert.Config(
    label="scene10",
    tag="cube0_base_box0",
    scene=OGScene10.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0"],
        ["cube0"],
        ["cube0", "box0"],
        ["box0"],
        ["box0"],
        ["box0", "ee_target"],
    ],
    segment_ids=[
        0,
        2,
        3,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        22,
    ],
    fix_bimodal=True,
)


cube1_base_base = TapasExpert.Config(
    label="scene10",
    tag="cube1_base_base",
    scene=OGScene10.Config(),
    gt_frames=[
        ["ee_init", "cube1"],
        ["cube1"],
        ["cube1"],
        ["cube1", "cube1_target"],
        ["cube1_target"],
        ["cube1_target"],
        ["cube1_target", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    fix_bimodal=True,
)

cube1_base_box1 = TapasExpert.Config(
    label="scene10",
    tag="cube1_base_box1",
    scene=OGScene10.Config(),
    gt_frames=[
        ["ee_init", "cube1"],
        ["cube1"],
        ["cube1"],
        ["cube1", "box1"],
        ["box1"],
        ["box1"],
        ["box1", "ee_target"],
    ],
    segment_ids=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21],
    fix_bimodal=True,
)

faucet0_a_b = TapasExpert.Config(
    label="scene10",
    tag="faucet0_a_b",
    scene=OGScene10.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0"],
        ["faucet0"],
        ["faucet0"],
        ["faucet0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22],
    snap_ee_actions=False,
)

faucet0_b_a = TapasExpert.Config(
    label="scene10",
    tag="faucet0_b_a",
    scene=OGScene10.Config(),
    gt_frames=[
        ["ee_init", "faucet0"],
        ["faucet0"],
        ["faucet0"],
        ["faucet0"],
        ["faucet0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)


agents = [
    lid0_base_base,
    lid0_base_box0,
    lid0_base_box1,
    cube0_base_base,
    cube0_base_box0,
    cube1_base_base,
    cube1_base_box1,
    faucet0_b_a,
    faucet0_a_b,
]
