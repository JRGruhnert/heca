import click


@click.command(help="Register a skill with a template.")
def register():
    click.echo("Skill registered!")
