from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene1 import OGScene1

faucet0_b_a = TapasExpert.Config(
    tag="faucet0_b_a",
    scene=OGScene1.Config(),
    gt_frames=[[]],
)

faucet1_b_a = TapasExpert.Config(
    tag="faucet1_b_a",
    scene=OGScene1.Config(),
    gt_frames=[[]],
)

cube0_base_base = TapasExpert.Config(
    tag="cube0_base_base",
    scene=OGScene1.Config(),
    gt_frames=[[]],
)

button0_s1_s0 = TapasExpert.Config(
    tag="button0_s1_s0",
    scene=OGScene1.Config(),
    gt_frames=[[]],
)

button0_s0_s1 = TapasExpert.Config(
    tag="button0_s0_s1",
    scene=OGScene1.Config(),
    gt_frames=[[]],
)

faucet1_a_b = TapasExpert.Config(
    tag="faucet1_a_b",
    scene=OGScene1.Config(),
    gt_frames=[[]],
)

cube0_base_shelf0 = TapasExpert.Config(
    tag="cube0_base_shelf0",
    scene=OGScene1.Config(),
    gt_frames=[[]],
)

faucet0_a_b = TapasExpert.Config(
    tag="faucet0_a_b",
    scene=OGScene1.Config(),
    gt_frames=[[]],
)


agents = [
    faucet0_b_a,
    faucet1_b_a,
    cube0_base_base,
    button0_s1_s0,
    button0_s0_s1,
    faucet1_a_b,
    cube0_base_shelf0,
    faucet0_a_b,
]
