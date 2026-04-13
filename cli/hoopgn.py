import click
from importlib.util import spec_from_file_location, module_from_spec


def config_handler(path, attr="cfg", configtype=None):
    spec = spec_from_file_location("dynamic_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the specified file: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = getattr(module, attr) if attr else module
    if configtype is not None:
        assert isinstance(
            cfg, configtype
        ), f"Config object must be of type {configtype}, got {type(cfg)}"
    return cfg


def hoopgn_config(func):
    return click.option(
        "--config",
        "-c",
        type=click.Path(exists=True),
        required=True,
        help="Path to config file",
    )(func)


@click.group()
@hoopgn_config
@click.pass_context
def hoopgn(ctx, cfg_path):
    ctx.ensure_object(dict)
    ctx.obj["hoopgn"] = cfg_path
