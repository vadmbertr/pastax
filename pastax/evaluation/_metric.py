import equinox as eqx


class Metric(eqx.Module):
    """
    Base class for metric objects.

    Attributes
    ----------
    metric_fun : str
        The name of the metric function or method.
    """

    metric_fun: str = eqx.field(static=True)


class LiuIndex(Metric):
    """
    The Liu index metric.

    Attributes
    ----------
    metric_fun : str
        The name of the metric function or method: `metric_fun="liu_index"`.
    """

    metric_fun: str = eqx.field(static=True, default_factory=lambda: "liu_index")


class Mae(Metric):
    """
    The Mean Absolute Error metric.

    Attributes
    ----------
    metric_fun : str
        The name of the metric function or method: `metric_fun="mae"`.
    """

    metric_fun: str = eqx.field(static=True, default_factory=lambda: "mae")


class Rmse(Metric):
    """
    The Root Mean Square Error metric.

    Attributes
    ----------
    metric_fun : str
        The name of the metric function or method: `metric_fun="rmse"`.
    """

    metric_fun: str = eqx.field(static=True, default_factory=lambda: "rmse")


class SeparationDistance(Metric):
    """
    The Separation distance metric.

    Attributes
    ----------
    metric_fun : str
        The name of the metric function or method: `metric_fun="separation_distance"`.
    """

    metric_fun: str = eqx.field(static=True, default_factory=lambda: "separation_distance")
