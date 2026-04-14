from dataclasses import dataclass
from hoopgn.observation.td_properties import TDProperties
from hoopgn.skills.skill import Skill, SkillConfig
import curses


@dataclass(kw_only=True)
class ManualSkillConfig(SkillConfig):
    skills: list[SkillConfig]


class ManualSkill(Skill):

    def __init__(
        self,
        config: ManualSkillConfig,
    ):
        super().__init__(config)
        self.config = config
        self.skills = [Skill(config=cfg) for cfg in config.skills]
        self.do_reset = False

    def act(
        self,
        obs: TDProperties,
        goal: TDProperties,
    ) -> Skill | None:
        choice = self.select_skill_tui()
        return self.skills[choice]

    def explain(self, current: TDProperties, goal: TDProperties, skill: Skill) -> str:
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
        def tui(stdscr):
            curses.curs_set(0)
            selected = 0
            while True:
                stdscr.clear()
                stdscr.addstr(0, 0, "Select a skill (arrow keys, Enter):")
                for idx, skill in enumerate(self.skills):
                    label = f"{skill.config.label}: {skill.config.description}"
                    if idx == selected:
                        stdscr.addstr(idx + 2, 2, label, curses.A_REVERSE)
                    else:
                        stdscr.addstr(idx + 2, 2, label)
                key = stdscr.getch()
                if key == curses.KEY_UP and selected > 0:
                    selected -= 1
                elif key == curses.KEY_DOWN and selected < len(skills) - 1:
                    selected += 1
                elif key in [curses.KEY_ENTER, 10, 13]:
                    return selected

        return curses.wrapper(tui)
