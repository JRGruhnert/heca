from hoopgn.evaluators.dense import DenseEvaluator
from hoopgn.evaluators.evaluator import Evaluator
from hoopgn.evaluators.sparse import SparseEvaluator


EVALUATOR_BUILDERS = {
    Evaluator.Config: lambda config: SparseEvaluator(config),
    DenseEvaluator.Config: lambda config: DenseEvaluator(config),
}


def register_evaluator(config_type, builder):
    EVALUATOR_BUILDERS[config_type] = builder


def select_evaluator(config: Evaluator.Config) -> Evaluator:
    builder = EVALUATOR_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in EVALUATOR_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config)
