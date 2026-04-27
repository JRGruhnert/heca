from typing import cast

from cli.hoopgn import config_handler
from heca.runners.skill.plot_skill_runner import (
    SkillPlotRunner,
    SkillPlotRunnerConfig,
)
import click


@click.command(help="Commands related to individual skills.")
@click.pass_context
def plot(ctx):
    cfg = cast(
        SkillPlotRunnerConfig,
        config_handler(path=ctx.obj["hoopgn"], configtype=SkillPlotRunnerConfig),
    )
    SkillPlotRunner(cfg).run()
