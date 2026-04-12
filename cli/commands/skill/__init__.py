import click
from importlib.util import spec_from_file_location, module_from_spec
from dataclasses import dataclass

from hoopgn.skills.skill import SkillConfig


@dataclass
class SkillCommandConfig:
    skills: list[SkillConfig]


@click.group()
def skill():
    pass


@skill.command(help="Commands related to individual skills.")
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to config file"
)
def explain(config):
    if config is None:
        raise click.UsageError("You must provide a --config path.")
    from hoopgn.runners.tps import run

    spec = spec_from_file_location("config", config)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the specified config file.")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    run(module.config)
