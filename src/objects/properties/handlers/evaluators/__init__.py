from src.objects.properties.handlers.evaluators.area_evaluator import (
    AreaEvaluator,
    AreaEvaluatorConfig,
)
from src.objects.properties.handlers.evaluators.ignore_evaluator import (
    IgnoreEvaluator,
    IgnoreEvaluatorConfig,
)
from src.objects.properties.handlers.evaluators.threshold_evaluator import (
    ThresholdEvaluator,
    ThresholdEvaluatorConfig,
)


STATE_EVALUATOR_BUILDERS = {
    AreaEvaluatorConfig: lambda config: AreaEvaluator(config),
    IgnoreEvaluatorConfig: lambda config: IgnoreEvaluator(config),
    ThresholdEvaluatorConfig: lambda config: ThresholdEvaluator(config),
}


def register_state_evaluator(config_type, builder):
    STATE_EVALUATOR_BUILDERS[config_type] = builder


def select_state_evaluator(config):
    builder = STATE_EVALUATOR_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in STATE_EVALUATOR_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
