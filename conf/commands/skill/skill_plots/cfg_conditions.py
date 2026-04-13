trained_skill_tag = "blue"
trained_state_tag = "blue"
checkpoint_name = f"t_{trained_skill_tag}_{trained_state_tag}_pe0.0_pr0.0"

skill_tag = "blue"
state_tag = "blue"

cfg = get_skill_plot_runner_config(
    skill_tag=skill_tag,
    property_tag=state_tag,
    plots=[
        SkillPlotConfig(
            plot_tag="d_blue_blue",
            checkpoint_name=checkpoint_name,
        )
    ],
)
