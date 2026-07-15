# heca/misc/policy_pool.py (NEW FILE)
import copy
from heca.misc.ppo import PPO
from heca.heca_gnn.network import Network
from heca.misc import logger


class PolicyPool:
    """Shared or private policy pool wrapping a PPO instance.

    pool_id="grasp"          → all agents with this ID share one PPO+network
    pool_id="heca_1_private" → only Heca-1 uses it

    Ensures a single inference copy that stays in sync after learn().
    """

    _instances: dict[str, "PolicyPool"] = {}

    def __init__(self, pool_id: str, ppo: PPO, network_cfg: Network.Config):
        self.pool_id = pool_id
        self.ppo = ppo
        self.inference_network: Network | None = None

    @classmethod
    def get_or_create(
        cls,
        pool_id: str,
        ppo_cfg: PPO.Config,
        network_cfg: Network.Config,
    ) -> "PolicyPool":
        if pool_id not in cls._instances:
            logger.info(f"Creating PolicyPool '{pool_id}'")
            ppo = PPO.get(ppo_cfg)
            ppo.setup(network_cfg)  # safe if we make setup idempotent (see below)
            pool = cls(pool_id, ppo, network_cfg)
            pool.inference_network = ppo.copy_network()
            cls._instances[pool_id] = pool
        return cls._instances[pool_id]

    def sync_inference(self):
        """Copy training weights → shared inference copy (all agents see it)."""
        if self.inference_network is not None and self.ppo.network is not None:
            self.inference_network.load_state_dict(self.ppo.network.state_dict())

    def learn(self, progress: float) -> tuple[dict, bool]:
        """PPO update, then sync the shared inference copy."""
        state_dict, is_best = self.ppo.learn(progress)
        self.sync_inference()
        return state_dict, is_best

    @classmethod
    def reset_all(cls):
        cls._instances.clear()
