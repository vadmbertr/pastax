"""
This module provides classes for evaluating simulated [`pastax.trajectory.Trajectory`][] and [`pastax.trajectory.TrajectoryEnsemble`][].
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
