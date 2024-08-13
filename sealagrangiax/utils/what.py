import equinox as eqx


class WHAT(eqx.Enumeration):
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
