from hoopgn.properties.features.evaluators.area_evaluator import (
    AreaEvaluator,
    AreaEvaluatorConfig,
)
from hoopgn.properties.features.evaluators.evaluator import (
    PropertyEvaluator,
    PropertyEvaluatorConfig,
)
from hoopgn.properties.features.evaluators.ignore_evaluator import (
    IgnoreEvaluator,
    IgnoreEvaluatorConfig,
)
from hoopgn.properties.features.evaluators.threshold_evaluator import (
    ThresholdEvaluator,
    ThresholdEvaluatorConfig,
)


_PROPERTY_EVALUATOR_BUILDERS = {
    AreaEvaluatorConfig: lambda config: AreaEvaluator(config),
    IgnoreEvaluatorConfig: lambda config: IgnoreEvaluator(config),
    ThresholdEvaluatorConfig: lambda config: ThresholdEvaluator(config),
}


def register_property_evaluator(config_type, builder):
    _PROPERTY_EVALUATOR_BUILDERS[config_type] = builder


def select_property_evaluator(config: PropertyEvaluatorConfig) -> PropertyEvaluator:
    builder = _PROPERTY_EVALUATOR_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in _PROPERTY_EVALUATOR_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
