import click
from importlib.util import spec_from_file_location, module_from_spec


@click.group()
def skill():
    """Commands related to skill learning."""
    pass


@skill.command()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to config file"
)
def explain(config):
    if config is None:
        raise click.UsageError("You must provide a --config path.")
    from cli.commands.skill.explain import entry_point

    spec = spec_from_file_location("config", config)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the specified config file.")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    entry_point(module.config)
