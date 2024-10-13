from typing import List

import equinox as eqx

from ..trajectory import TrajectoryEnsemble, Trajectory
from ._evaluation import Evaluation
from ._metric import LiuIndex, Mae, Metric, Rmse, SeparationDistance


class Evaluator(eqx.Module):
    """
    Base class for evaluating trajectories using a set of predefined metrics.
    
    Attributes
    -----------
        metrics (List[Metric]): A list of metrics used for evaluation. The default metrics are the separation distance,
            the Liu Index, the MAE, and the RMSE.

    Methods
    -------
    __call__(self, reference_trajectory: Trajectory, simulated_trajectory: Trajectory | TrajectoryEnsemble) -> Evaluation
        Evaluates the `simulated_trajectory` (which might be an ensemble of trajectories) against the
        `reference_trajectory` using the defined metrics.
    """

    metrics: List[Metric] = eqx.field(default_factory=lambda: [SeparationDistance(), LiuIndex(), Mae(), Rmse()])

    @eqx.filter_jit
    def __call__(
        self, 
        reference_trajectory: Trajectory, 
        simulated_trajectory: Trajectory | TrajectoryEnsemble
    ) -> Evaluation:
        """
        Evaluates the `simulated_trajectory` (which might be an ensemble of trajectories) against the
        `reference_trajectory` using the defined metrics.

        Parameters
        ----------
            reference_trajectory (Trajectory): The reference trajectory to compare against.
            simulated_trajectory (Trajectory | TrajectoryEnsemble): The simulated trajectory or ensemble of trajectories
                to be evaluated.

        Returns
        -------
            Evaluation: The result of the evaluation.

        Raises
        ------
            NotImplementedError: This method should be implemented by child classes.
        """
        raise NotImplementedError
