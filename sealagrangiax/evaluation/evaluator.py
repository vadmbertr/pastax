import equinox as eqx

from ..trajectory import Timeseries, TimeseriesEnsemble, Trajectory, TrajectoryEnsemble
from ..utils import WHAT
from ._evaluation import Evaluation
from ._evaluator import Evaluator
from ._metric import METRIC


class PairEvaluator(Evaluator):
    @eqx.filter_jit
    def __call__(
            self,
            reference_trajectory: Trajectory,
            simulated_trajectory: Trajectory
    ) -> [Timeseries, ...]:
        print("Computing evaluation metrics...")

        metrics = {}
        for metric in self.metrics:
            metric_name = METRIC[metric.metric]

            values = getattr(reference_trajectory, metric_name)(simulated_trajectory)
            metrics[metric_name] = Timeseries(
                values, reference_trajectory.times, metric.what, metric.unit
            )

        return Evaluation(metrics)


class EnsembleEvaluator(Evaluator):
    @eqx.filter_jit
    def __call__(
            self,
            reference_trajectory: Trajectory,
            simulated_trajectories: TrajectoryEnsemble
    ) -> [Timeseries | TimeseriesEnsemble, ...]:
        print("Computing evaluation metrics...")

        metrics = {}
        for metric in self.metrics:
            metric_name = METRIC[metric.metric]

            ensemble = getattr(simulated_trajectories, metric_name)(reference_trajectory)
            ensemble = TimeseriesEnsemble(
                ensemble, reference_trajectory.times, metric.what, metric.unit
            )

            crps = simulated_trajectories.crps(
                reference_trajectory, distance_func=getattr(Trajectory, metric_name)
            )
            crps = Timeseries(
                crps, reference_trajectory.times, WHAT.crps, metric.unit
            )

            mean = ensemble.mean()
            mean = Timeseries(
                mean, reference_trajectory.times, WHAT.mean, metric.unit
            )

            metrics[metric_name] = (ensemble, crps, mean)

        return Evaluation(metrics)
