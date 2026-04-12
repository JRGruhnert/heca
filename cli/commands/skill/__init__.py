import click
from importlib.util import spec_from_file_location, module_from_spec
from dataclasses import dataclass

from conf.entities.skills.tapas_skills import TapasSkillSet
from hoopgn.skills.skill import SkillConfig


@click.group()
def skill():
    pass


@skill.command(help="Commands related to individual skills.")
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Path to config file"
)
def plot(config):
    if config is None:
        raise click.UsageError("You must provide a --config path.")
    from hoopgn.runners.skill.plot_skill_runner import SkillPlotter, SkillPlotterConfig

    spec = spec_from_file_location("config", config)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the specified config file.")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    skills = [
        v for k, v in TapasSkillSet.__dict__.items() if isinstance(v, SkillConfig)
    ]
    entities = []  # TODO:
    properties = []  # TODO:
    plots = []  # TODO:
    SkillPlotter(
        SkillPlotterConfig(
            skills=skills,
            entities=entities,
            properties=properties,
            plots=plots,
        )
    )()
