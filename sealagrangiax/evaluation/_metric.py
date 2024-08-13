import equinox as eqx

from ..utils import UNIT, WHAT


class METRIC(eqx.Enumeration):
    liu_index = "liu_index"
    mae = "mae"
    rmse = "rmse"
    separation_distance = "separation_distance"


class Metric(eqx.Module):
    metric: METRIC = eqx.field(static=True)
    what: WHAT = eqx.field(static=True)
    unit: UNIT = eqx.field(static=True)


class LiuIndex(Metric):  # noqa
    metric: METRIC = eqx.field(static=True, default_factory=lambda: METRIC.liu_index)
    what: WHAT = eqx.field(static=True, default_factory=lambda: WHAT.liu_index)
    unit: UNIT = eqx.field(static=True, default_factory=lambda: UNIT.dimensionless)


class MAE(Metric):
    metric: METRIC = eqx.field(static=True, default_factory=lambda: METRIC.mae)
    what: WHAT = eqx.field(static=True, default_factory=lambda: WHAT.mae)
    unit: UNIT = eqx.field(static=True, default_factory=lambda: UNIT.meters)


class RMSE(Metric):  # noqa
    metric: METRIC = eqx.field(static=True, default_factory=lambda: METRIC.rmse)
    what: WHAT = eqx.field(static=True, default_factory=lambda: WHAT.rmse)
    unit: UNIT = eqx.field(static=True, default_factory=lambda: UNIT.meters)


class SeparationDistance(Metric):
    metric: METRIC = eqx.field(static=True, default_factory=lambda: METRIC.separation_distance)
    what: WHAT = eqx.field(static=True, default_factory=lambda: WHAT.separation_distance)
    unit: UNIT = eqx.field(static=True, default_factory=lambda: UNIT.meters)
