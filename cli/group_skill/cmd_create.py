import click


@click.command(help="Create a skill with a template.")
def create():
    click.echo("Skill created!")
