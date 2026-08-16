from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene3 import OGScene3

button2_s1_s0 = TapasExpert.Config(
    tag="button2_s1_s0",
    scene=OGScene3.Config(),
    gt_frames=[[]],
)

button1_s0_s1 = TapasExpert.Config(
    tag="button1_s0_s1",
    scene=OGScene3.Config(),
    gt_frames=[[]],
)

button2_s0_s1 = TapasExpert.Config(
    tag="button2_s0_s1",
    scene=OGScene3.Config(),
    gt_frames=[[]],
)

cube0_base_base = TapasExpert.Config(
    tag="cube0_base_base",
    scene=OGScene3.Config(),
    gt_frames=[[]],
)

cube0_base_shelf0 = TapasExpert.Config(
    tag="cube0_base_shelf0",
    scene=OGScene3.Config(),
    gt_frames=[[]],
)

button1_s1_s0 = TapasExpert.Config(
    tag="button1_s1_s0",
    scene=OGScene3.Config(),
    gt_frames=[[]],
)

drawer0_b_a = TapasExpert.Config(
    tag="drawer0_b_a",
    scene=OGScene3.Config(),
    gt_frames=[[]],
)

button0_s1_s0 = TapasExpert.Config(
    tag="button0_s1_s0",
    scene=OGScene3.Config(),
    gt_frames=[[]],
)

drawer0_a_b = TapasExpert.Config(
    tag="drawer0_a_b",
    scene=OGScene3.Config(),
    gt_frames=[[]],
)

cube0_base_drawer0 = TapasExpert.Config(
    tag="cube0_base_drawer0",
    scene=OGScene3.Config(),
    gt_frames=[[]],
)


agents = [
    button2_s1_s0,
    button1_s0_s1,
    button2_s0_s1,
    cube0_base_base,
    cube0_base_shelf0,
    button1_s1_s0,
    drawer0_b_a,
    button0_s1_s0,
    drawer0_a_b,
    cube0_base_drawer0,
]
