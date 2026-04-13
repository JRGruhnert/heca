from typing import cast

import click

from cli.hoopgn import config_handler
from hoopgn.runners.train_runner import TrainRunner, TrainRunnerConfig


@click.command()
@click.pass_context
def train(ctx):
    cfg_path = ctx.obj["hoopgn"]
    cfg = cast(
        TrainRunnerConfig,
        config_handler(path=cfg_path, configtype=TrainRunnerConfig),
    )
    TrainRunner(cfg).run()
