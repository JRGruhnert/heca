from cli.heca import heca
from cli.cmd_eval import eval
from .cmd_explain import explain
from .cmd_help import help
from .cmd_plot import plot
from .cmd_train import train
from .group_skill import skill

heca.add_command(eval)
heca.add_command(explain)
heca.add_command(help)
heca.add_command(plot)
heca.add_command(train)
heca.add_command(skill)


def entry_point():
    heca()
