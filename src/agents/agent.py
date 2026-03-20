from abc import ABC, abstractmethod
from src.observation.observation import StateValueDict
from src.skills.skill import Skill



class Agent(ABC):

    @abstractmethod
    def act(
        self,
        obs: StateValueDict,
        goal: StateValueDict,
    ) -> Skill:
        """Select an action given the current observation and goal observation."""
        raise NotImplementedError("Act method not implemented yet.")

    @abstractmethod
    def feedback(self, reward: float, success: bool, terminal: bool) -> bool:
        """Pass feedback from the environment. Returns True if the buffer reached the targeted batch size."""
        raise NotImplementedError("Feedback method not implemented yet.")

    @abstractmethod
    def learn(self) -> bool:
        """Perform learning update. Returns True if training should stop. (Plateau reached)"""
        raise NotImplementedError("Learn method not implemented yet.")

    @abstractmethod
    def save(self, tag: str = ""):
        raise NotImplementedError("Save method not implemented yet.")

    @abstractmethod
    def load(self):
        """Load model from checkpoint path."""
        raise NotImplementedError("Load method not implemented yet.")
    
    @abstractmethod
    def metadata(self) -> dict:
        """Return agent metadata as a dictionary."""
        raise NotImplementedError("Metadata method not implemented yet.")

    @abstractmethod
    def metrics(self) -> dict:
        """Return current agent metrics as a dictionary."""
        raise NotImplementedError("Metrics method not implemented yet.")
