from typing import cast

from cli.hoopgn import config_handler

import click

from heca.runners.plotter import HecaPlotter


@click.command(help="Commands related to individual skills.")
@click.pass_context
def plot(ctx):
    cfg = cast(
        HecaPlotter.Config,
        config_handler(path=ctx.obj["hoopgn"], configtype=HecaPlotter.Config),
    )
    HecaPlotter(cfg).run()
