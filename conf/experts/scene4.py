from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene4 import OGScene4

peg0_base_base = TapasExpert.Config(
    tag="peg0_base_base",
    scene=OGScene4.Config(),
    gt_frames=[[]],
)

lid0_base_base = TapasExpert.Config(
    tag="lid0_base_base",
    scene=OGScene4.Config(),
    gt_frames=[[]],
)

cube0_base_base = TapasExpert.Config(
    tag="cube0_base_base",
    scene=OGScene4.Config(),
    gt_frames=[[]],
)

cube0_base_shelf0 = TapasExpert.Config(
    tag="cube0_base_shelf0",
    scene=OGScene4.Config(),
    gt_frames=[[]],
)

cube0_base_box0 = TapasExpert.Config(
    tag="cube0_base_box0",
    scene=OGScene4.Config(),
    gt_frames=[[]],
)

lid0_base_box0 = TapasExpert.Config(
    tag="lid0_base_box0",
    scene=OGScene4.Config(),
    gt_frames=[[]],
)


agents = [
    peg0_base_base,
    lid0_base_base,
    cube0_base_base,
    cube0_base_shelf0,
    cube0_base_box0,
    lid0_base_box0,
]
