from cli.group_skill.skill import skill
from cli.group_skill.cmd_create import create
from cli.group_skill.cmd_plot import plot
from cli.group_skill.cmd_register import register
from cli.group_skill.cmd_train import train

skill.add_command(create)
skill.add_command(plot)
skill.add_command(register)
skill.add_command(train)
