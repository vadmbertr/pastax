import equinox as eqx

from ..trajectory import TrajectoryEnsemble, Trajectory
from ._evaluation import Evaluation
from ._metric import LiuIndex, Mae, Metric, Rmse, SeparationDistance


class Evaluator(eqx.Module):  # TODO: should it be an eqx Module?
    """
    Base class for evaluating trajectories using a set of predefined metrics.
    
    Attributes
    -----------
    metrics : list[Metric]
        A list of [`pastax.evaluation.Metric`][]s used for evaluation. 
        The default [`pastax.evaluation.Metric`][]s are [`pastax.evaluation.SeparationDistance`][], [`pastax.evaluation.LiuIndex`][], [`pastax.evaluation.Mae`][], 
        and [`pastax.evaluation.Rmse`][].

    Methods
    -------
    __call__(self, reference_trajectory, simulated_trajectory)
        Evaluates the `simulated_trajectory` (which might be an ensemble of trajectories) 
        against the `reference_trajectory` using `self.metrics`.
    """

    metrics: list[Metric] = eqx.field(default_factory=lambda: [SeparationDistance(), LiuIndex(), Mae(), Rmse()])

    @eqx.filter_jit
    def __call__(
        self, 
        reference_trajectory: Trajectory, 
        simulated_trajectory: Trajectory | TrajectoryEnsemble
    ) -> Evaluation:
        """
        Evaluates the `simulated_trajectory` (which might be an ensemble of trajectories) 
        against the `reference_trajectory` using `self.metrics`.

        Parameters
        ----------
        reference_trajectory : Trajectory
            The reference [`pastax.trajectory.Trajectory`][] to compare against.
        simulated_trajectory : Trajectory | TrajectoryEnsemble
            The simulated [`pastax.trajectory.Trajectory`][] or [`pastax.trajectory.TrajectoryEnsemble`][] to be evaluated.

        Returns
        -------
        Evaluation
            The result of the [`pastax.evaluation.Evaluation`][].

        Raises
        ------
        NotImplementedError
            This method should be implemented by child classes.
        """
        raise NotImplementedError
