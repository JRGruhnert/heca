from src.evaluators.dense import DenseEvaluator, DenseEvaluatorConfig
from src.evaluators.dense2 import Dense2Evaluator, Dense2EvaluatorConfig
from src.evaluators.dense3 import Dense3Evaluator, Dense3EvaluatorConfig
from src.evaluators.evaluator import Evaluator, EvaluatorConfig
from src.evaluators.sparse import SparseEvaluator, SparseEvaluatorConfig


EVALUATOR_BUILDERS = {
    SparseEvaluatorConfig: lambda config: SparseEvaluator(config),
    DenseEvaluatorConfig: lambda config: DenseEvaluator(config),
    Dense2EvaluatorConfig: lambda config: Dense2Evaluator(config),
    Dense3EvaluatorConfig: lambda config: Dense3Evaluator(config),
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
