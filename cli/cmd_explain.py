from typing import cast

import click

from cli.hoopgn import config_handler
from heca.runners.explainer import ExplainRunner, ExplainRunnerConfig


@click.command()
@click.pass_context
def explain(ctx):
    cfg_path = ctx.obj["hoopgn"]
    cfg = cast(
        ExplainRunnerConfig,
        config_handler(path=cfg_path, configtype=ExplainRunnerConfig),
    )
    ExplainRunner(cfg).run()
