from typing import cast

import click

from cli.hoopgn import config_handler
from heca.runners.runner import HecaRunner


@click.command()
@click.pass_context
def explain(ctx):
    cfg_path = ctx.obj["hoopgn"]
    cfg = cast(
        HecaRunner.Config,
        config_handler(path=cfg_path, configtype=HecaRunner.Config),
    )
    HecaRunner(cfg).explain()
