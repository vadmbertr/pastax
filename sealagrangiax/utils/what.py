import equinox as eqx


class WHAT(eqx.Enumeration):
    """
    JAX-compatible enumeration class that categorizes various types of measurements and metrics.

    Attributes
    ----------
    unknown : WHAT
        An unknown type.
        
    location : WHAT
        Position in [latitude, longitude].
    displacement : WHAT
        Displacement in [latitude, longitude].
    time : WHAT
        Time since epoch.
        
    earth_distance : WHAT
        Distance on earth.
    spatial_length : WHAT
        Spatial length.
    spatial_step : WHAT
        Spatial step.
        
    euclidean_distance : WHAT
        Euclidean distance.
    liu_index : WHAT
        Liu index.
    mae : WHAT
        Mean Absolute Error (MAE).
    rmse : WHAT
        Root Mean Square Error (RMSE).
    separation_distance : WHAT
        Separation distance.
        
    crps : WHAT
        Continuous Ranked Probability Score (CRPS).
    ensemble_dispersion : WHAT
        Ensemble dispersion.
    mean : WHAT
        Mean value.
    """
    unknown = ""

    location = "Position in [latitude, longitude]"
    displacement = "Displacement in [latitude, longitude]"
    time = "Time since epoch"

    earth_distance = "Distance on earth"
    spatial_length = "Spatial length"
    spatial_step = "Spatial step"

    euclidean_distance = "Euclidean distance"
    liu_index = "Liu index"
    mae = "MAE"
    rmse = "RMSE"
    separation_distance = "Separation distance"

    crps = "CRPS"
    ensemble_dispersion = "Ensemble dispersion"
    mean = "Mean"
