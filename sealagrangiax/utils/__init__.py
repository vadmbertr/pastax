"""
This module provides various geographical and unit conversion and manipulation utilities in JAX.
"""


from .unit import (
    Unit, Dimensionless,
    Meters, Kilometers, LatLonDegrees,
    Seconds, Minutes, Hours, Days,
    UNIT,
    degrees_to_meters, degrees_to_kilometers, 
    meters_to_degrees, meters_to_kilometers, 
    kilometers_to_degrees, kilometers_to_meters, 
    time_in_seconds, seconds_to_days,
    units_to_str, compose_units,
)
from .geo import EARTH_RADIUS, distance_on_earth, longitude_in_180_180_degrees


__all__ = [
    "EARTH_RADIUS", "distance_on_earth", "longitude_in_180_180_degrees", 
    "Unit", "Dimensionless", 
    "Meters", "Kilometers", "LatLonDegrees",
    "Seconds", "Minutes", "Hours", "Days", 
    "UNIT",
    "degrees_to_meters", "degrees_to_kilometers", 
    "meters_to_degrees", "meters_to_kilometers", 
    "kilometers_to_degrees", "kilometers_to_meters", 
    "time_in_seconds", "seconds_to_days",
    "units_to_str", "compose_units",
]
