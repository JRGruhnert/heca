import click


@click.group()
def hoopgn():
    """Main CLI for hoopgn project."""
    pass


@hoopgn.command()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to config file"
)
def train(config):
    from cli.commands.train import entry_point

    entry_point(config)


@hoopgn.command()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to config file"
)
def explain(config):
    from cli.commands.explain import entry_point

    entry_point(config)


if __name__ == "__main__":
    hoopgn()


def cli_entry_point():
    hoopgn()
