"""pastax: Differentiable Lagrangian simulator for ocean surface trajectories."""

from ._safe_math import safe_divide, safe_log, safe_sqrt
from .forcing import Dataset, Field
from .grid import Grid
from .geo import (
    EARTH_RADIUS,
    degrees_to_meters,
    haversine,
    meters_to_degrees,
)
from .interpolation import bilinear_interp_2d, linear_interp_1d, spatiotemporal_interp
from .metric import liu_index, normalized_separation_distance, separation_distance
from .score import (
    dawid_sebastiani,
    energy_score,
    squared_error,
    variogram_score,
)
from .solver import (
    RK4,
    Dopri5,
    Euler,
    EulerHeun,
    Heun,
    ItoMilstein,
    StratonovichMilstein,
    Tsit5,
    solve,
)

__version__ = "0.1.0"

__all__ = [
    # safe math
    "safe_sqrt",
    "safe_log",
    "safe_divide",
    # geo
    "EARTH_RADIUS",
    "haversine",
    "meters_to_degrees",
    "degrees_to_meters",
    # interpolation
    "linear_interp_1d",
    "bilinear_interp_2d",
    "spatiotemporal_interp",
    # forcing
    "Field",
    "Dataset",
    "Grid",
    # solver
    "Euler",
    "Heun",
    "RK4",
    "Tsit5",
    "Dopri5",
    "EulerHeun",
    "ItoMilstein",
    "StratonovichMilstein",
    "solve",
    # metric
    "separation_distance",
    "normalized_separation_distance",
    "liu_index",
    # score
    "squared_error",
    "dawid_sebastiani",
    "energy_score",
    "variogram_score",
]
