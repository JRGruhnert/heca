from dataclasses import dataclass
from hoopgn.agents.agent import Agent, AgentConfig
from hoopgn.buffer import Buffer
from hoopgn.storage import Storage
from hoopgn.observation.observation import StateValueDict
from hoopgn.skills.skill import Skill


@dataclass(kw_only=True)
class HumanAgentConfig(AgentConfig):
    pass


class HumanAgent(Agent):

    def __init__(
        self,
        config: HumanAgentConfig,
        storage: Storage,
    ):
        self.config = config
        self.storage = storage
        self.do_reset = False

    def act(
        self,
        obs: StateValueDict,
        goal: StateValueDict,
    ) -> Skill | None:
        """Select an action given the current observation and goal observation."""
        for i, skill in enumerate(self.storage.skills):
            print(f"{i}: {skill.config.label}")
        print(f"{len(self.storage.skills)}: Reset")
        choice = int(input("Enter the Task id: "))
        if choice == len(self.storage.skills):
            self.do_reset = True
            return None
        return self.storage.skills[choice]

    def explain(
        self, current: StateValueDict, goal: StateValueDict, skill: Skill
    ) -> str:
        raise NotImplementedError("")

    def feedback(self, reward: float, success: bool, terminal: bool) -> bool:
        """Pass feedback from the environment. Returns True if the buffer reached the targeted batch size."""
        if self.do_reset:
            print("Resetting agent...")
            self.do_reset = False
            return True
        return False

    def learn(self) -> bool:
        """Perform learning update. Returns True if training should stop. (Plateau reached)"""
        return False

    def save(self, tag: str = ""):
        pass

    def load(self):
        pass

    def metadata(self) -> dict:
        """Return agent metadata as a dictionary."""
        return {}

    def metrics(self) -> dict:
        """Return current agent metrics as a dictionary."""
        return {}
