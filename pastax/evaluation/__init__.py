"""
This module provides classes for evaluating simulated [`pastax.Trajectory`][] and [`pastax.TrajectoryEnsemble`][].
"""


from ._evaluation import Evaluation
from ._evaluator import Evaluator
from .evaluator import EnsembleEvaluator, PairEvaluator
from ._metric import Metric, LiuIndex, Mae, Rmse, SeparationDistance

__all__ = [
    "Evaluation",
    "Evaluator",
    "EnsembleEvaluator", "PairEvaluator",
    "Metric", "LiuIndex", "Mae", "Rmse", "SeparationDistance",
]
