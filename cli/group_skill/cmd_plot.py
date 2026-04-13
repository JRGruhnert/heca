from typing import cast

from cli.hoopgn import config_handler
from hoopgn.runners.skill.plot_skill_runner import (
    SkillPlotRunner,
    SkillPlotRunnerConfig,
)
import click

from hoopgn.skills.skill import Skill, SkillConfig


@click.command(help="Commands related to individual skills.")
@click.pass_context
def plot(ctx):
    hoopgn_cfg = cast(
        SkillPlotRunnerConfig,
        config_handler(path=ctx.obj["hoopgn"], configtype=SkillPlotRunnerConfig),
    )
    skill_cfg = cast(
        SkillConfig,
        config_handler(path=ctx.obj["skill"], configtype=SkillConfig),
    )
    SkillPlotRunner(hoopgn_cfg).run(Skill(skill_cfg))
