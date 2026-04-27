from typing import cast

import click

from cli.hoopgn import config_handler
from heca.runners.plotter import PlotRunner, PlotRunnerConfig


@click.command()
@click.pass_context
def plot(ctx):
    cfg_path = ctx.obj["hoopgn"]
    cfg = cast(
        PlotRunnerConfig,
        config_handler(path=cfg_path, configtype=PlotRunnerConfig),
    )
    PlotRunner(cfg).run()
