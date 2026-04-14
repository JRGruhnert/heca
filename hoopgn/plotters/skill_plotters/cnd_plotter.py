from dataclasses import dataclass

from conf.entities.properties.area import CalvinAreaConfig
from hoopgn.observation.td_entity import TDEntity
from hoopgn.plotters.plots.entity_3d import (
    Entity3DHelper,
    Entity3DHelperConfig,
    Entity3DHelperConfig,
    Entity3DMode,
    EntityPoint,
)
from hoopgn.plotters.skill_plotters.skill_plotter import (
    SkillPlotter,
    SkillPlotterConfig,
)
from hoopgn.properties.features.conditions.condition import PropertyCondition
from hoopgn.properties.states.area_state import AreaStateConfig
from hoopgn.skills.skill import Skill


@dataclass
class SkillConditionsPlotterConfig(SkillPlotterConfig):
    title: str = "Skill Conditions"
    subdir: str = "conditions"
    name: str = "SkillConditions3D"
    entity3d: Entity3DHelperConfig = Entity3DHelperConfig(
        mode=Entity3DMode.COLORS,
    )
    area: AreaStateConfig = CalvinAreaConfig()


class SkillConditionsPlot(SkillPlotter):
    def __init__(self, config: SkillConditionsPlotterConfig):
        super().__init__(config)
        self.config = config
        self.entity_dict = {e.config.label: e for e in self.entities}
        self.entity3d = Entity3DHelper(config.entity3d)

    def plot_content(self, skill: Skill):
        for entity in self.entities:
            label = entity.config.label
            self.entity3d.set_entity(
                entity=entity,
                current=self.make_td_entity(label, skill.precons),
                goal=self.make_td_entity(label, skill.postcons),
                different=True,  # Makes no sense otherwise
                solved=False,  # We dont support that currently
            )
        self.entity3d.show_entities()
        self.entity3d.show_areas(self.config.area)
        self.entity3d.finish()

    def make_td_entity(
        self, label: str, conditions: dict[str, PropertyCondition]
    ) -> TDEntity:
        if label in ["red", "blue", "pink"]:
            label = f"block_{label}"
        if label in ["slider", "drawer", "button", "led", "lightbulb"]:
            label = f"base__{label}"
        return TDEntity(
            position=conditions[f"{label}_position"].value,
            rotation=conditions[f"{label}_rotation"].value,
            state=conditions[f"{label}_scalar"].value,
        )
