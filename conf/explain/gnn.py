from conf.explain.common import get_explain_config
from src.states.states import get_number_of_states_by_tag
from src.skills.skills import get_number_of_skills_by_tag
from src.variables import SET_BLUE, SET_PINK, SET_RED, SET_SLIDE

trained_skill_tag = SET_SLIDE
trained_state_tag = SET_SLIDE
checkpoint_name = f"t_{trained_skill_tag}_{trained_state_tag}_pe0.0_pr0.0"

skill_tag = SET_SLIDE
state_tag = SET_SLIDE

config = get_explain_config(
    skill_count=get_number_of_skills_by_tag(skill_tag),
    state_count=get_number_of_states_by_tag(state_tag),
    skill_set_tag=skill_tag,
    state_set_tag=state_tag,
    is_gnn=True,
    prefix_tag="d",
    checkpoint_name=checkpoint_name,
)
