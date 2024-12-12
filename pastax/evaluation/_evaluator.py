import equinox as eqx

from ..trajectory import Trajectory, TrajectoryEnsemble
from ._evaluation import Evaluation
from ._metric import LiuIndex, Mae, Metric, Rmse, SeparationDistance


class BaseEvaluator(eqx.Module):  # TODO: should it be an eqx Module?
    """
    Base class for evaluating trajectories using a set of predefined metrics.

    Attributes
    -----------
    metrics : list[Metric]
        A list of [`pastax.evaluation.Metric`][]s used for evaluation.
        The default [`pastax.evaluation.Metric`][]s are [`pastax.evaluation.SeparationDistance`][],
        [`pastax.evaluation.LiuIndex`][], [`pastax.evaluation.Mae`][], and [`pastax.evaluation.Rmse`][].

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
        simulated_trajectory: Trajectory | TrajectoryEnsemble,
    ) -> Evaluation:
        """
        Evaluates the `simulated_trajectory` (which might be an ensemble of trajectories)
        against the `reference_trajectory` using `self.metrics`.

        Parameters
        ----------
        reference_trajectory : Trajectory
            The reference [`pastax.trajectory.Trajectory`][] to compare against.
        simulated_trajectory : Trajectory | TrajectoryEnsemble
            The simulated [`pastax.trajectory.Trajectory`][] or [`pastax.trajectory.TrajectoryEnsemble`][] to be
            evaluated.

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


class PairEvaluator(BaseEvaluator):
    """
    Class for evaluating a simulated trajectory using a set of predefined metrics.

    Methods
    -------
    __call__(self, reference_trajectory, simulated_trajectory)
        Evaluates the `simulated_trajectory` against the `reference_trajectory` using the `self.metrics`.
    """

    def __call__(self, reference_trajectory: Trajectory, simulated_trajectory: Trajectory) -> Evaluation:
        """
        Evaluates the `simulated_trajectory` against the `reference_trajectory` using `self.metrics`.

        Parameters
        ----------
        reference_trajectory : Trajectory
            The reference [`pastax.trajectory.Trajectory`][] to compare against.
        simulated_trajectory : Trajectory
            The simulated [`pastax.trajectory.Trajectory`][] to be evaluated.

        Returns
        -------
        Evaluation
            The result of the [`pastax.evaluation.Evaluation`][].
        """
        metrics = {}
        for metric in self.metrics:
            metric_fun = metric.metric_fun
            metrics[metric_fun] = getattr(reference_trajectory, metric_fun)(simulated_trajectory)

        return Evaluation(metrics)


class EnsembleEvaluator(BaseEvaluator):
    """
    Class for evaluating an ensemble of simulated trajectories using a set of predefined metrics.

    Methods
    -------
    __call__(self, reference_trajectory, simulated_trajectory)
        Evaluates the `simulated_trajectories` ensemble against the `reference_trajectory` using `self.metrics`.
    """

    def __call__(
        self,
        reference_trajectory: Trajectory,
        simulated_trajectories: TrajectoryEnsemble,
    ) -> Evaluation:
        """
        Evaluates the `simulated_trajectories` ensemble against the `reference_trajectory` using `self.metrics`.

        Parameters
        ----------
        reference_trajectory : Trajectory
            The reference [`pastax.trajectory.Trajectory`][] to compare against.
        simulated_trajectories : TrajectoryEnsemble
            The simulated [`pastax.trajectory.TrajectoryEnsemble`][] to be evaluated.

        Returns
        -------
        Evaluation
            The result of the [`pastax.evaluation.Evaluation`][].
        """
        metrics = {}
        for metric in self.metrics:
            metric_fun = metric.metric_fun

            ensemble = getattr(simulated_trajectories, metric_fun)(reference_trajectory)
            ensemble = getattr(simulated_trajectories, metric_fun)(reference_trajectory)

            crps = simulated_trajectories.crps(reference_trajectory, metric_func=getattr(Trajectory, metric_fun))

            mean = ensemble.mean(axis=0)

            metrics[metric_fun] = (ensemble, crps, mean)

        return Evaluation(metrics)
