import equinox as eqx

from ..timeseries import Timeseries, TimeseriesEnsemble, Trajectory, TrajectoryEnsemble
from ..utils import WHAT
from ._evaluation import Evaluation
from ._evaluator import Evaluator
from ._metric import METRIC_FUN


class PairEvaluator(Evaluator):
    """
    Class for evaluating a simulated trajectory using a set of predefined metrics.

    Methods
    -------
    __call__(self, reference_trajectory: Trajectory, simulated_trajectory: Trajectory) -> Evaluation
        Evaluates the `simulated_trajectory` against the `reference_trajectory` using the defined metrics.
    """
    
    @eqx.filter_jit
    def __call__(self, reference_trajectory: Trajectory, simulated_trajectory: Trajectory) -> Evaluation:
        """
        Evaluates the `simulated_trajectory` against the `reference_trajectory` using the defined metrics.
        
        Parameters
        ----------
            reference_trajectory (Trajectory): The reference trajectory to compare against.
            simulated_trajectory (Trajectory): The simulated trajectory to be evaluated.
        
        Returns
        -------
            Evaluation: The result of the evaluation.
        """
        metrics = {}
        for metric in self.metrics:
            metric_name = METRIC_FUN[metric.metric_fun]

            values = getattr(reference_trajectory, metric_name)(simulated_trajectory)
            metrics[metric_name] = Timeseries(
                values, reference_trajectory.times, metric.what, metric.unit
            )

        return Evaluation(metrics)


class EnsembleEvaluator(Evaluator):
    """
    Class for evaluating an ensemble of simulated trajectories using a set of predefined metrics.

    Methods
    -------
    __call__(self, reference_trajectory: Trajectory, simulated_trajectory: TrajectoryEnsemble) -> Evaluation
        Evaluates the `simulated_trajectories` ensemble against the `reference_trajectory` using the defined metrics.
    """

    @eqx.filter_jit
    def __call__(self, reference_trajectory: Trajectory, simulated_trajectories: TrajectoryEnsemble) -> Evaluation:
        """
        Evaluates the `simulated_trajectories` ensemble against the `reference_trajectory` using the defined metrics.
        
        Parameters
        ----------
            reference_trajectory (Trajectory): The reference trajectory to compare against.
            simulated_trajectory (Trajectory): The simulated ensemble of trajectories to be evaluated.
        
        Returns
        -------
            Evaluation: The result of the evaluation.
        """
        metrics = {}
        for metric in self.metrics:
            metric_fun = METRIC_FUN[metric.metric_fun]

            ensemble = getattr(simulated_trajectories, metric_fun)(reference_trajectory)
            ensemble = TimeseriesEnsemble(
                ensemble, reference_trajectory.times, metric.what, metric.unit
            )

            crps = simulated_trajectories.crps(
                reference_trajectory, distance_func=getattr(Trajectory, metric_fun)
            )
            crps = Timeseries(
                crps, reference_trajectory.times, WHAT.crps, metric.unit
            )

            mean = ensemble.mean()
            mean = Timeseries(
                mean, reference_trajectory.times, WHAT.mean, metric.unit
            )

            metrics[metric_fun] = (ensemble, crps, mean)

        return Evaluation(metrics)
