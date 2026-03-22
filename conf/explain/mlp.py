from conf.explain.common import (
    get_quick_config,
)

checkpoint_name = "rf_srpb_srp_pe0.0_pr0.0"
prefix = "s"
skill_tag = "sr"
state_tag = "srpb"

config = get_quick_config(
    skill_set_tag=skill_tag,
    state_set_tag=state_tag,
    is_baseline=True,
    prefix_tag=prefix,
    checkpoint_name=checkpoint_name,
)
