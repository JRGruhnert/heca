from hoopgn.evaluators.dense import DenseEvaluator, DenseEvaluatorConfig
from hoopgn.evaluators.dense2 import Dense2Evaluator, Dense2EvaluatorConfig
from hoopgn.evaluators.dense3 import Dense3Evaluator, Dense3EvaluatorConfig
from hoopgn.evaluators.evaluator import Evaluator, EvaluatorConfig
from hoopgn.evaluators.set_evaluator import SetEvaluator, SetEvaluatorConfig
from hoopgn.evaluators.sparse import SparseEvaluator, SparseEvaluatorConfig


EVALUATOR_BUILDERS = {
    SparseEvaluatorConfig: lambda config: SparseEvaluator(config),
    DenseEvaluatorConfig: lambda config: DenseEvaluator(config),
    Dense2EvaluatorConfig: lambda config: Dense2Evaluator(config),
    Dense3EvaluatorConfig: lambda config: Dense3Evaluator(config),
    SetEvaluatorConfig: lambda config: SetEvaluator(config),
}


def register_evaluator(config_type, builder):
    EVALUATOR_BUILDERS[config_type] = builder


def select_evaluator(config: EvaluatorConfig) -> Evaluator:
    builder = EVALUATOR_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in EVALUATOR_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
