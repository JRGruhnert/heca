import click
import questionary
from cli.commands.skill import skill
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

CONFIG_ROOT = Path("conf/commands")


def _get_file_description(file_path: Path) -> str:
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith('"""') or line.startswith("'''"):
                    return line.strip("\"' ")
                if line.startswith("#"):
                    return line.lstrip("# ")
    except Exception:
        pass
    return ""


def _pick_config(config_dir: Path) -> str | None:
    configs = sorted(p for p in config_dir.glob("cfg_*.py"))
    if configs:
        choices = [
            questionary.Choice(
                title=f"{p.name}", value=str(p), description=_get_file_description(p)
            )
            for p in configs
        ]
        return questionary.select("Select config:", choices=choices).ask()
    else:
        click.echo(f"No config files found in {config_dir}")
        return None


def _find_config_dir(path: list[str]) -> Path | None:
    for i in range(len(path), 0, -1):
        candidate = CONFIG_ROOT / "/".join(path[:i])
        if candidate.is_dir():
            return candidate
    return None


def _interactive_walk(group: click.Group, path: list[str]):
    commands = group.list_commands(click.Context(group))
    if not commands:
        return

    choices = []
    for sub_label in commands:
        sub = group.get_command(click.Context(group), sub_label)
        if sub:
            choices.append(
                questionary.Choice(
                    title=sub_label,
                    value=sub_label,
                    description=sub.help,
                )
            )
    choices.append(
        questionary.Choice(title="Quit", value=None, description="Exit this menu.")
    )

    chosen = questionary.select("Select:", choices=choices).ask()
    if chosen:
        cmd = group.get_command(click.Context(group), chosen)
        if cmd:
            new_path = path + [chosen]

            if isinstance(cmd, click.Group):
                _interactive_walk(cmd, new_path)
            elif isinstance(cmd, click.Command):
                config_dir = _find_config_dir(new_path)
                if config_dir:
                    config_path = _pick_config(config_dir)
                    if config_path:
                        ctx = click.Context(cmd, parent=click.Context(group))
                        ctx.invoke(cmd, config=config_path)
                    else:
                        click.echo("No config selected.")
                else:
                    click.echo(f"No config directory found for command: {new_path[-1]}")
            else:
                click.echo(f"Unknown command type for: {chosen}")
        elif chosen == "Quit":
            click.echo("Exiting.")
        else:
            click.echo(f"Command not found: {chosen}")
    else:
        click.echo("Exiting.")


@click.group(invoke_without_command=True)
@click.pass_context
def hoopgn(ctx):
    if ctx.invoked_subcommand is None:
        _interactive_walk(hoopgn, [])


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


def entry_point():
    hoopgn()
