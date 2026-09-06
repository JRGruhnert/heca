from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene7 import OGScene7

cube1_base_base = TapasExpert.Config(
    label="scene7",
    tag="cube1_base_base",
    scene=OGScene7.Config(),
    gt_frames=[
        ["ee_init", "cube1"],
        ["cube1"],
        ["cube1"],
        ["cube1", "cube1_target"],
        ["cube1_target"],
        ["cube1_target"],
        ["cube1_target", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21],
    fix_bimodal=True,
    pos_only=False,
)

cube1_base_shelf0 = TapasExpert.Config(
    label="scene7",
    tag="cube1_base_shelf0",
    scene=OGScene7.Config(),
    gt_frames=[
        ["ee_init", "cube1"],
        ["cube1"],
        ["cube1"],
        ["cube1", "shelf0"],
        ["shelf0"],
        ["shelf0"],
        ["shelf0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    fix_bimodal=True,
    pos_only=False,
)

cube0_base_base = TapasExpert.Config(
    label="scene7",
    tag="cube0_base_base",
    scene=OGScene7.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0"],
        ["cube0"],
        ["cube0", "cube0_target"],
        ["cube0_target"],
        ["cube0_target"],
        ["cube0_target", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20],
    fix_bimodal=True,
    pos_only=False,
)

cube0_base_box0 = TapasExpert.Config(
    label="scene7",
    tag="cube0_base_box0",
    scene=OGScene7.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0"],
        ["cube0"],
        ["cube0", "box0"],
        ["box0"],
        ["box0"],
        ["box0", "ee_target"],
    ],
    segment_ids=[0, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22],
    fix_bimodal=True,
    pos_only=False,
)

lid0_base_base = TapasExpert.Config(
    label="scene7",
    tag="lid0_base_base",
    scene=OGScene7.Config(),
    gt_frames=[
        ["ee_init", "lid0"],
        ["lid0"],
        ["lid0"],
        ["lid0", "lid0_target"],
        ["lid0_target"],
        ["lid0_target"],
        ["lid0_target", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    fix_bimodal=True,
    pos_only=False,
)


lid0_base_box0 = TapasExpert.Config(
    label="scene7",
    tag="lid0_base_box0",
    scene=OGScene7.Config(),
    gt_frames=[
        ["ee_init", "lid0"],
        ["lid0"],
        ["lid0"],
        ["lid0", "box0"],
        ["box0"],
        ["box0"],
        ["box0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    fix_bimodal=True,
    pos_only=False,
)

agents = [
    cube1_base_base,
    cube1_base_shelf0,
    cube0_base_base,
    cube0_base_box0,
    lid0_base_base,
    lid0_base_box0,
]
