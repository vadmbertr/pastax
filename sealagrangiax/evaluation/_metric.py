import equinox as eqx

from ..utils import UNIT, WHAT


class METRIC_FUN(eqx.Enumeration):
    """
    Enumeration of metric function names.

    Members
    -------
    liu_index: Represents the Liu Index metric function (or method) name.
    mae: Represents the Mean Absolute Error metric function (or method) name.
    rmse: Represents the Root Mean Square Error metric function (or method) name.
    separation_distance: Represents the Separation Distance metric function (or method) name.
    """

    liu_index = "liu_index"
    mae = "mae"
    rmse = "rmse"
    separation_distance = "separation_distance"


class Metric(eqx.Module):
    """
    Base class for metric objects.

    Attributes
    ----------
    metric_fun (METRIC_FUN): The metric function name.
    what (WHAT): The long name of the metric.
    unit (UNIT): The unit of the metric.
    """

    metric_fun: METRIC_FUN = eqx.field(static=True)
    what: WHAT = eqx.field(static=True)
    unit: UNIT = eqx.field(static=True)


class LiuIndex(Metric):
    """
    The Liu Index metric.

    Attributes
    ----------
    metric_fun (METRIC_FUN): The metric function name, defaulting to METRIC_FUN.liu_index.
    what (WHAT): The long name of the metric, defaulting to WHAT.liu_index.
    unit (UNIT): The unit of the metric, defaulting to UNIT.dimensionless.
    """
    
    metric_fun: METRIC_FUN = eqx.field(static=True, default_factory=lambda: METRIC_FUN.liu_index)
    what: WHAT = eqx.field(static=True, default_factory=lambda: WHAT.liu_index)
    unit: UNIT = eqx.field(static=True, default_factory=lambda: UNIT.dimensionless)


class Mae(Metric):
    """
    The Mean Absolute Error metric.

    Attributes
    ----------
    metric_fun (METRIC_FUN): The metric function name, defaulting to METRIC_FUN.mae.
    what (WHAT): The long name of the metric, defaulting to WHAT.mae.
    unit (UNIT): The unit of the metric, defaulting to UNIT.meters.
    """
    
    metric_fun: METRIC_FUN = eqx.field(static=True, default_factory=lambda: METRIC_FUN.mae)
    what: WHAT = eqx.field(static=True, default_factory=lambda: WHAT.mae)
    unit: UNIT = eqx.field(static=True, default_factory=lambda: UNIT.meters)


class Rmse(Metric):
    """
    The Root Mean Square Error metric.

    Attributes
    ----------
    metric_fun (METRIC_FUN): The metric function name, defaulting to METRIC_FUN.rmse.
    what (WHAT): The long name of the metric, defaulting to WHAT.rmse.
    unit (UNIT): The unit of the metric, defaulting to UNIT.meters.
    """
    
    metric_fun: METRIC_FUN = eqx.field(static=True, default_factory=lambda: METRIC_FUN.rmse)
    what: WHAT = eqx.field(static=True, default_factory=lambda: WHAT.rmse)
    unit: UNIT = eqx.field(static=True, default_factory=lambda: UNIT.meters)


class SeparationDistance(Metric):
    """
    The Separation Distance metric.

    Attributes
    ----------
    metric_fun (METRIC_FUN): The metric function name, defaulting to METRIC_FUN.separation_distance.
    what (WHAT): The long name of the metric, defaulting to WHAT.separation_distance.
    unit (UNIT): The unit of the metric, defaulting to UNIT.meters.
    """
    
    metric_fun: METRIC_FUN = eqx.field(static=True, default_factory=lambda: METRIC_FUN.separation_distance)
    what: WHAT = eqx.field(static=True, default_factory=lambda: WHAT.separation_distance)
    unit: UNIT = eqx.field(static=True, default_factory=lambda: UNIT.meters)
