import sys
import numpy as np
import tapas_gmm_modified
from dataclasses import dataclass
from functools import cached_property
from tapas_gmm_modified.utils.robot_trajectory import RobotTrajectory
from tapas_gmm_modified.policy.gmm import GMMPolicy, GMMPolicyConfig
from tapas_gmm_modified.policy.models.tpgmm import AutoTPGMM

from heca.agents.agent import AgentFeedback, Cursor
from heca.agents.experts.expert import ExpertAgent
from heca.entities.entity import Entity
from heca.environment.scenes.scene import Scene
from heca.environment.world import MetaWorld
from heca.misc.td import TDEntity, TDScene, TDWorld
from heca.misc import logger
from heca.misc.hardware import device

sys.modules["tapas_gmm"] = tapas_gmm_modified  # alias for unpickling old checkpoints


class TapasAgent(ExpertAgent):
    @dataclass(kw_only=True)
    class Config(ExpertAgent.Config):
        policy: GMMPolicyConfig

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.state = Cursor.IDLE
        self.scene: Scene = MetaWorld.get(self.cfg.scene)

    def act(self, x: TDScene, y: TDScene) -> tuple[TDScene, AgentFeedback]:
        self.policy.reset_episode()
        predictions = self.make_prediction(x, y)
        if predictions is None:
            return x, AgentFeedback(
                reward=0.0,
                done=True,
                terminal=True,
                state=Cursor.ERROR,
                learn=False,
            )
        self.state = Cursor.ACTIVE

        while not predictions.is_finished:
            pred = predictions.step()
            action = np.concatenate((pred.ee, pred.gripper))  # type: ignore
            z = self.scene.step(action)

        # TODO: compute actual reward, done, terminal based on scene evaluation
        reward, done = self.evaluate(z, y)
        return z, AgentFeedback(
            reward=reward,
            done=done,
            terminal=True,
            state=self.state,
            learn=False,
        )

    def make_prediction(self, x: TDWorld, y: TDWorld) -> RobotTrajectory | None:
        tx = x.get(self.cfg.scene.label)["tapas"]
        # ty = y.get(self.cfg.scene.label)["tapas"]
        try:
            prds, _ = self.policy.predict(tx)  # type: ignore
            return prds  # type: ignore
        except FloatingPointError as e:
            logger.debug(f"Numerical error in prediction: {e}")
            return None
        except Exception as e:
            logger.debug(f"General error in prediction: {e}")
            return None

    def from_disk(self, path: str):
        logger.info(f"Loading tapas policy from: {path}")
        temp = GMMPolicy(self.cfg.policy)
        assert isinstance(temp, GMMPolicy), "Policy model must be a GMMPolicy."
        temp.from_disk(path)
        temp.eval()
        self.policy = temp.to(device)

    @cached_property
    def model(self) -> AutoTPGMM:
        temp = self.policy.model
        assert isinstance(temp, AutoTPGMM), "Model must be an AutoTPGMM."
        return temp

    def gen_pre(self) -> list[tuple[Entity, TDEntity]]:
        raise NotImplementedError()

    def gen_post(self) -> list[tuple[Entity, TDEntity]]:
        raise NotImplementedError()
