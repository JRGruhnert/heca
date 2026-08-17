from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene4 import OGScene4

peg0_base_base = TapasExpert.Config(
    tag="peg0_base_base",
    scene=OGScene4.Config(),
    gt_frames=[
        ["ee_init", "peg0"],
        ["peg0", "peg0_target"],
        ["peg0_target", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)

lid0_base_base = TapasExpert.Config(
    tag="lid0_base_base",
    scene=OGScene4.Config(),
    gt_frames=[
        ["ee_init", "lid0"],
        ["lid0", "lid0_target"],
        ["lid0_target", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)

cube0_base_base = TapasExpert.Config(
    tag="cube0_base_base",
    scene=OGScene4.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0", "cube0_target"],
        ["cube0_target", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)

cube0_base_shelf0 = TapasExpert.Config(
    tag="cube0_base_shelf0",
    scene=OGScene4.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0", "shelf0"],
        ["shelf0", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)

cube0_base_box0 = TapasExpert.Config(
    tag="cube0_base_box0",
    scene=OGScene4.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0", "box0"],
        ["box0", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)

lid0_base_box0 = TapasExpert.Config(
    tag="lid0_base_box0",
    scene=OGScene4.Config(),
    gt_frames=[
        ["ee_init", "lid0"],
        ["lid0", "box0"],
        ["box0", "ee_target"],
    ],
    segment_ids=[0, 1, 3, 4, 5],
)


agents = [
    peg0_base_base,
    lid0_base_base,
    cube0_base_base,
    cube0_base_shelf0,
    cube0_base_box0,
    lid0_base_box0,
]
