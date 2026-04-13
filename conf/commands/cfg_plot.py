from conf.entity_set import get_entity_set
from hoopgn.runners.plot_runner import PlotRunnerConfig


from conf.property_sets import get_property_set
from conf.skill_sets import get_skill_set


SKILLS = "blue"
PROPERTIES_EVAL = "blue"
PROPERTIES_NETWORK = "blue"
ENTITIES = "blue"


skills = get_skill_set(SKILLS)
entities = get_entity_set(ENTITIES)
properties_eval = get_property_set(PROPERTIES_EVAL)
properties_network = get_property_set(PROPERTIES_NETWORK)
properties = properties_eval
plots = []  # TODO: add plots

cfg = PlotRunnerConfig(
    skills=skills,
    entities=entities,
    properties=properties,
    plots=plots,
)
