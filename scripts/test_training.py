from heca.agents.experts.tapas import TapasAgent

# from heca.agents.heca import Heca
from heca.agents.heca import Heca
from heca.learning.ppo import PPO
from heca.learning.buffers.fair_buffer import FairBuffer
from heca.scenes.ogbench.scene import OGBenchScene

heca_cfg = Heca.Config(
    tag="test",
    agents=[
        TapasAgent.Config(
            tag="open_window",
            scene=OGBenchScene.Config(),
            use_gt=True,
        ),
        TapasAgent.Config(
            tag="close_window",
            scene=OGBenchScene.Config(),
            use_gt=True,
        ),
        TapasAgent.Config(
            tag="lock_right_button",
            scene=OGBenchScene.Config(),
            use_gt=True,
        ),
        TapasAgent.Config(
            tag="unlock_right_button",
            scene=OGBenchScene.Config(),
            use_gt=True,
        ),
    ],
    learner=PPO.Config(
        tag="test_virt_norm",
        # buffer=APPOBuffer.Config(),
        buffer=FairBuffer.Config(),
        normalize_rewards=True,
    ),
    downstream_virtual=True,
)

heca = Heca.get(heca_cfg)
heca.train(OGBenchScene.Config())
