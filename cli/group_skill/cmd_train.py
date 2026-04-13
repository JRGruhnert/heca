import click


@click.command(help="Train a skill with a template.")
def train():
    click.echo("Skill trained!")
