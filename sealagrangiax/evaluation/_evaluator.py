import equinox as eqx

from ..trajectory import Timeseries, TimeseriesEnsemble, Trajectory
from ._metric import LiuIndex, MAE, Metric, RMSE, SeparationDistance


class Evaluator(eqx.Module):
    metrics: [Metric, ...] = eqx.field(default_factory=lambda: [SeparationDistance(), LiuIndex(), MAE(), RMSE()])

    @eqx.filter_jit
    def __call__(
            self,
            reference_trajectory: Trajectory,
            simulated_trajectory: Trajectory
    ) -> [Timeseries | TimeseriesEnsemble, ...]:
        raise NotImplementedError
