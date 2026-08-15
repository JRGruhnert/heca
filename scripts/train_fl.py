from heca.agents.heca import Heca
from heca.learning.fppo import FPPO
from heca.learning.server import FLServer
from heca.scenes.ogbench.scene1 import OGScene1

server = FLServer.Config(
    label="global"
)
exp1 = [
    Heca.Config(
        tag="heca1",
        agents=[],
        learner=FPPO.Config(tag="heca1"),
        scene=OGScene1.Config(),
    ),
]


def train(self):
    """Train the network with PPO for a given number of episodes."""
    agents = [Heca.get(cfg) for cfg in exp1]

        x, y = self.sample()
        # print("Starting Episode")
        z = self.act(x, y)  # runs a full episode to terminal, accumulates PPO data
        # print("Ending Episode")
