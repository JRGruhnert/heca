from conf.explain.common import get_explain_config


trained_skill_tag = "slider"
trained_state_tag = "slider"
checkpoint_name = f"t_{trained_skill_tag}_{trained_state_tag}_pe0.0_pr0.0"

skill_tag = "slider"
state_tag = "slider"

config = get_explain_config(
    skill_set_tag=skill_tag,
    state_set_tag=state_tag,
    is_gnn=True,
    prefix_tag="d",
    checkpoint_name=checkpoint_name,
)
