from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene4 import OGScene4

peg0_base_base = TapasExpert.Config(
    label="scene4",
    tag="peg0_base_base",
    scene=OGScene4.Config(),
    gt_frames=[
        ["ee_init", "peg0"],
        ["peg0"],
        ["peg0"],
        ["peg0", "peg0_target"],
        ["peg0_target"],
        ["peg0_target"],
        ["peg0_target", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20],
    fix_bimodal=True,
    pos_only=False,
)

cube0_base_shelf0 = TapasExpert.Config(
    label="scene4",
    tag="cube0_base_shelf0",
    scene=OGScene4.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0"],
        ["cube0"],
        ["cube0", "shelf0"],
        ["shelf0"],
        ["shelf0"],
        ["shelf0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20],
    fix_bimodal=True,
    pos_only=False,
)

cube0_base_base = TapasExpert.Config(
    label="scene4",
    tag="cube0_base_base",
    scene=OGScene4.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0"],
        ["cube0"],
        ["cube0", "cube0_target"],
        ["cube0_target"],
        ["cube0_target"],
        ["cube0_target", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 20, 21, 22],
    fix_bimodal=True,
    pos_only=False,
)

cube0_base_box0 = TapasExpert.Config(
    label="scene4",
    tag="cube0_base_box0",
    scene=OGScene4.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0"],
        ["cube0"],
        ["cube0", "box0"],
        ["box0"],
        ["box0"],
        ["box0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20],
    fix_bimodal=True,
    pos_only=False,
)

lid0_base_base = TapasExpert.Config(
    label="scene4",
    tag="lid0_base_base",
    scene=OGScene4.Config(),
    gt_frames=[
        ["ee_init", "lid0"],
        ["lid0"],
        ["lid0"],
        ["lid0", "lid0_target"],
        ["lid0_target"],
        ["lid0_target"],
        ["lid0_target", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    fix_bimodal=True,
    pos_only=False,
)

lid0_base_box0 = TapasExpert.Config(
    label="scene4",
    tag="lid0_base_box0",
    scene=OGScene4.Config(),
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


agents = [
    peg0_base_base,
    lid0_base_base,
    cube0_base_base,
    cube0_base_shelf0,
    cube0_base_box0,
    lid0_base_box0,
]
