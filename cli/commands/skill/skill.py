import click
from cli.hoopgn import hoopgn


@hoopgn.group()
def skill():
    """Commands related to skill learning."""
    pass


@skill.command()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to config file"
)
def explain(config):
    from cli.commands.skill.explain import entry_point

    entry_point(config)
