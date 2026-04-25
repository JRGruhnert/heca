from dataclasses import dataclass
from hoopgn.agents.hoop import BranchAgent
from hoopgn.observation import TDProperties
from hoopgn.agents.agent import Agent
import curses


class HumanAgent(BranchAgent):
    @dataclass(kw_only=True)
    class Config(BranchAgent.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.do_reset = False

    def act(
        self,
        obs: TDProperties,
        goal: TDProperties,
    ) -> Agent | None:
        choice = self.select_skill_tui()
        return Agent.search(choice)

    def explain(self, current: TDProperties, goal: TDProperties, skill: Agent) -> str:
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

    def select_skill_tui(self):
        def tui(stdscr) -> Agent.Signature:
            curses.curs_set(0)
            selected = 0
            while True:
                stdscr.clear()
                stdscr.addstr(0, 0, "Select a skill (arrow keys, Enter):")
                for idx, agent in enumerate(list(self.cfg.agents)):
                    label = f"{agent.label}: {agent.description}"
                    if idx == selected:
                        stdscr.addstr(idx + 2, 2, label, curses.A_REVERSE)
                    else:
                        stdscr.addstr(idx + 2, 2, label)
                key = stdscr.getch()
                if key == curses.KEY_UP and selected > 0:
                    selected -= 1
                elif key == curses.KEY_DOWN and selected < len(self.cfg.agents) - 1:
                    selected += 1
                elif key in [curses.KEY_ENTER, 10, 13]:
                    return list(self.cfg.agents)[selected]

        return curses.wrapper(tui)
