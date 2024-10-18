import equinox as eqx


class Metric(eqx.Module):
    """
    Base class for metric objects.

    Attributes
    ----------
    metric_fun : str 
        The name of the metric function or method.
    name : str 
        The name of the metric.
    """

    metric_fun: str = eqx.field(static=True)


class LiuIndex(Metric):
    """
    The Liu index metric.

    Attributes
    ----------
    metric_fun : str
        The name of the metric function or method (set to "liu_index").
    name : str
        The name of the metric (set to "Liu index").
    """
    
    metric_fun: str = eqx.field(static=True, default_factory=lambda: "liu_index")


class Mae(Metric):
    """
    The Mean Absolute Error metric.

    Attributes
    ----------
    metric_fun : str
        The name of the metric function or method (set to "mae").
    name : str
        The name of the metric (set to "MAE").
    """
    
    metric_fun: str = eqx.field(static=True, default_factory=lambda: "mae")


class Rmse(Metric):
    """
    The Root Mean Square Error metric.

    Attributes
    ----------
    metric_fun : str
        The name of the metric function or method (set to "rmse").
    name : str
        The name of the metric (set to "RMSE").
    """
    
    metric_fun: str = eqx.field(static=True, default_factory=lambda: "rmse")


class SeparationDistance(Metric):
    """
    The Separation distance metric.

    Attributes
    ----------
    metric_fun : str
        The name of the metric function or method (set to "separation_distance").
    name : str
        The name of the metric (set to "Separation distance").
    """
    
    metric_fun: str = eqx.field(static=True, default_factory=lambda: "separation_distance")
