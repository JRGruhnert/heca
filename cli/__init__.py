from cli.hoopgn import hoopgn
from cli.cmd_eval import eval
from .cmd_explain import explain
from .cmd_help import help
from .cmd_plot import plot
from .cmd_sweep import sweep
from .cmd_train import train
from .group_skill import skill


hoopgn.add_command(eval)
hoopgn.add_command(explain)
hoopgn.add_command(help)
hoopgn.add_command(plot)
hoopgn.add_command(sweep)
hoopgn.add_command(train)
hoopgn.add_command(skill)


def entry_point():
    hoopgn()
