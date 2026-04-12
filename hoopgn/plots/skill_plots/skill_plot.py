from dataclasses import dataclass
import re

from hoopgn.plots.plot import Plot, PlotConfig
from hoopgn.skills.skill import Skill


@dataclass
class SkillPlotConfig(PlotConfig):
    networks: list[str]
    file_pattern: re.Pattern
    tag_pattern: re.Pattern


class SkillPlot(Plot):
    def __init__(self, config: SkillPlotConfig):
        self.config = config

    def __call__(self, skill: Skill):
        pass
