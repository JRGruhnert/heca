import click
from cli.commands.skill.skill import skill
from importlib.util import spec_from_file_location, module_from_spec


@click.group()
def hoopgn():
    """Main CLI for hoopgn project."""
    pass


hoopgn.add_command(skill)


@hoopgn.command()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to config file"
)
def train(config):
    if config is None:
        raise click.UsageError("You must provide a --config path.")
    from cli.commands.train import entry_point

    spec = spec_from_file_location("config", config)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the specified config file.")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    entry_point(module.config)


@hoopgn.command()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to config file"
)
def explain(config):
    if config is None:
        raise click.UsageError("You must provide a --config path.")
    from cli.commands.explain import entry_point

    spec = spec_from_file_location("config", config)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the specified config file.")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    entry_point(module.config)


if __name__ == "__main__":
    hoopgn()


def cli_entry_point():
    hoopgn()
