"""
This module provides various geographical and [`pastax.utils.Unit`][] conversion and manipulation utilities in JAX.
"""

from ._geo import distance_on_earth, EARTH_RADIUS, longitude_in_180_180_degrees
from ._unit import (
    compose_units,
    Days,
    degrees_to_kilometers,
    degrees_to_meters,
    Hours,
    Kilometers,
    kilometers_to_degrees,
    kilometers_to_meters,
    LatLonDegrees,
    Meters,
    meters_to_degrees,
    meters_to_kilometers,
    Minutes,
    Seconds,
    seconds_to_days,
    time_in_seconds,
    UNIT,
    Unit,
    units_to_str,
)


__all__ = [
    "EARTH_RADIUS",
    "distance_on_earth",
    "longitude_in_180_180_degrees",
    "Unit",
    "Meters",
    "Kilometers",
    "LatLonDegrees",
    "Seconds",
    "Minutes",
    "Hours",
    "Days",
    "UNIT",
    "degrees_to_meters",
    "degrees_to_kilometers",
    "meters_to_degrees",
    "meters_to_kilometers",
    "kilometers_to_degrees",
    "kilometers_to_meters",
    "time_in_seconds",
    "seconds_to_days",
    "units_to_str",
    "compose_units",
]
