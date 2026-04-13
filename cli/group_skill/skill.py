import click


def skill_config(func):
    return click.option(
        "--skill",
        "-s",
        type=click.Path(exists=True),
        required=True,
        help="Path to skill config file",
    )(func)


@click.group()
@skill_config
@click.pass_context
def skill(ctx, cfg_path):
    ctx.ensure_object(dict)
    ctx.obj["skill"] = cfg_path
