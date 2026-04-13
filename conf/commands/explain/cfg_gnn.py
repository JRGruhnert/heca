from conf.commands.explain.common import get_plot_runner_config


trained_skill_tag = "blue"
trained_state_tag = "blue"
checkpoint_name = f"t_{trained_skill_tag}_{trained_state_tag}_pe0.0_pr0.0"

skill_tag = "blue"
state_tag = "blue"

config = get_plot_runner_config(
    skill_set_tag=skill_tag,
    state_set_tag=state_tag,
    is_gnn=True,
    prefix_tag="d",
    checkpoint_name=checkpoint_name,
)
