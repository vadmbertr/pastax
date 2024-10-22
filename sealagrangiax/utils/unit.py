from __future__ import annotations
from fractions import Fraction
import functools as ft
from typing import Literal, Dict

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float    
import numpy as np

from .geo import distance_on_earth, EARTH_RADIUS


ft.total_ordering
class Unit(eqx.Module):
    """
    A base class for units of measurement.

    Attributes
    ----------
    name : str
        The name of the unit.

    Methods
    -------
    __eq__(other)
        Checks if two units are equal.
    __lt__(other)
        Checks if one unit is less than another.
    __hash__()
        Returns the hash of the unit.
    __repr__()
        Returns the string representation of the unit.
    convert_to(unit, value, exp, *args)
        Converts the value to the specified unit.
    """

    name: str = eqx.field(static=True, default_factory=lambda: "")

    def __eq__(self, other):
        if isinstance(other, Unit):
            return self.name == other.name
        return False

    def __lt__(self, other):
        if isinstance(other, Unit):
            return self.name < other.name
        return NotImplemented
    
    def __hash__(self):
        return hash(self.name)
    
    def __repr__(self):
        return self.name
    
    @staticmethod
    def _pre_convert(value: ArrayLike, exp: int | float) -> ArrayLike:
        """
        Prepares the value for conversion between "base" units by raising it to the power of the reciprocal of the exponent.

        Parameters
        ----------
        value : ArrayLike
            The value to be converted.
        exp : int or float
            The exponent to use for conversion.

        Returns
        -------
        ArrayLike
            The prepared value.
        """
        if exp == 1:
            return value
        return value ** (1 / exp)
    
    @staticmethod
    def _post_convert(value: ArrayLike, exp: int | float) -> ArrayLike:
        """
        Finalizes the conversion between "base" units by raising the value to the power of the exponent.

        Parameters
        ----------
        value : ArrayLike
            The value to be converted.
        exp : int or float
            The exponent to use for conversion.

        Returns
        -------
        ArrayLike
            The converted value.
        """
        if exp == 1:
            return value
        return value ** exp
    
    def convert_to(self, unit: Unit, value: ArrayLike, exp: int | float = 1, *args) -> ArrayLike:
        """
        Converts the value to the specified unit.

        Parameters
        ----------
        unit : Unit
            The unit to convert to.
        value : ArrayLike
            The value to convert.
        exp : int or float, optional
            The exponent to use for conversion (default is 1).
        *args
            Additional arguments for conversion.

        Returns
        -------
        ArrayLike
            The converted value.

        Raises
        ------
        NotImplementedError
            If not implemented by subclasses.
        """
        raise NotImplementedError
    

class Dimensionless(Unit):
    """
    A class representing a dimensionless unit.
    """

    def convert_to(self, unit: Unit, value: ArrayLike, exp: int | float = 1, *args) -> ArrayLike:
        """
        Converts the value to the specified unit.

        Parameters
        ----------
        unit : Unit
            The unit to convert to.
        value : ArrayLike
            The value to convert.
        exp : int or float, optional
            The exponent to use for conversion (default is 1).
        *args
            Additional arguments for conversion.

        Returns
        -------
        ArrayLike
            The converted value.
        """
        return value


class Meters(Unit):
    """
    A class representing meters as a unit of measurement.

    Attributes
    ----------
    name : str
        The name of the unit (set to "m").
    """

    name: str = eqx.field(static=True, default_factory=lambda: "m")

    def convert_to(self, unit: Unit, value: ArrayLike, exp: int | float = 1, *args) -> ArrayLike:
        """
        Converts the value to the specified unit.

        Parameters
        ----------
        unit : Unit
            The unit to convert to.
        value : ArrayLike
            The value to convert.
        exp : int or float, optional
            The exponent to use for conversion (default is 1).
        *args
            Additional arguments for conversion.

        Returns
        -------
        ArrayLike
            The converted value.

        Raises
        ------
        ValueError
            If the conversion is not possible.
        """
        if isinstance(unit, Meters):
            return value
        
        value = self._pre_convert(value, exp)

        if isinstance(unit, Kilometers):
            value = meters_to_kilometers(value)
        elif isinstance(unit, LatLonDegrees):
            value = meters_to_degrees(value, *args)
        else:
            raise ValueError(f"Cannot convert {self} to {unit}")
    
        return self._post_convert(value, exp)


class Kilometers(Unit):
    """
    A class representing kilometers as a unit of measurement.

    Attributes
    ----------
    name : str
        The name of the unit (set to "km").
    """

    name: str = eqx.field(static=True, default_factory=lambda: "km")

    def convert_to(self, unit: Unit, value: ArrayLike, exp: int | float = 1, *args) -> ArrayLike:
        """
        Converts the value to the specified unit.

        Parameters
        ----------
        unit : Unit
            The unit to convert to.
        value : ArrayLike
            The value to convert.
        exp : int or float, optional
            The exponent to use for conversion (default is 1).
        *args
            Additional arguments for conversion.

        Returns
        -------
        ArrayLike
            The converted value.

        Raises
        ------
        ValueError
            If the conversion is not possible.
        """
        if isinstance(unit, Kilometers):
            return value
        
        value = self._pre_convert(value, exp)

        if isinstance(unit, Meters):
            value = kilometers_to_meters(value)
        elif isinstance(unit, LatLonDegrees):
            value = kilometers_to_degrees(value, *args)
        else:
            raise ValueError(f"Cannot convert {self} to {unit}")
    
        return self._post_convert(value, exp)


class LatLonDegrees(Unit):
    """
    A class representing latitude and longitude degrees as a unit of measurement.

    Attributes
    ----------
    name : str
        The name of the unit (set to "°").
    """

    name: str = eqx.field(static=True, default_factory=lambda: "°")

    def convert_to(self, unit: Unit, value: ArrayLike, exp: int | float = 1, *args) -> ArrayLike:
        """
        Converts the value to the specified unit.

        Parameters
        ----------
        unit : Unit
            The unit to convert to.
        value : ArrayLike
            The value to convert.
        exp : int or float, optional
            The exponent to use for conversion (default is 1).
        *args
            Additional arguments for conversion.

        Returns
        -------
        ArrayLike
            The converted value.

        Raises
        ------
        ValueError
            If the conversion is not possible.
        """
        if isinstance(unit, LatLonDegrees):
            return value

        value = self._pre_convert(value, exp)

        if isinstance(unit, Meters):
            value = degrees_to_meters(value, *args)
        elif isinstance(unit, Kilometers):
            value = degrees_to_kilometers(value, *args)
        else:
            raise ValueError(f"Cannot convert {self} to {unit}")
    
        return self._post_convert(value, exp)


class Seconds(Unit):
    """
    A class representing seconds as a unit of measurement.

    Attributes
    ----------
    name : str
        The name of the unit (set to "s").
    """

    name: str = eqx.field(static=True, default_factory=lambda: "s")

    def convert_to(self, unit: Unit, value: ArrayLike, exp: int | float = 1, *args) -> ArrayLike:
        """
        Converts the value to the specified unit.

        Parameters
        ----------
        unit : Unit
            The unit to convert to.
        value : ArrayLike
            The value to convert.
        exp : int or float, optional
            The exponent to use for conversion (default is 1).
        *args
            Additional arguments for conversion.

        Returns
        -------
        ArrayLike
            The converted value.

        Raises
        ------
        ValueError
            If the conversion is not possible.
        """
        if isinstance(unit, Seconds):
            return value
        
        value = self._pre_convert(value, exp)

        if isinstance(unit, Minutes):
            value = seconds_to_minutes(value)
        elif isinstance(unit, Hours):
            value = seconds_to_hours(value)
        elif isinstance(unit, Days):
            value = seconds_to_days(value)
        else:
            raise ValueError(f"Cannot convert {self} to {unit}")
    
        return self._post_convert(value, exp)


class Minutes(Unit):
    """
    A class representing minutes as a unit of measurement.

    Attributes
    ----------
    name : str
        The name of the unit (set to "min").
    """

    name: str = eqx.field(static=True, default_factory=lambda: "min")

    def convert_to(self, unit: Unit, value: ArrayLike, exp: int | float = 1, *args) -> ArrayLike:
        """
        Converts the value to the specified unit.

        Parameters
        ----------
        unit : Unit
            The unit to convert to.
        value : ArrayLike
            The value to convert.
        exp : int or float, optional
            The exponent to use for conversion (default is 1).
        *args
            Additional arguments for conversion.

        Returns
        -------
        ArrayLike
            The converted value.

        Raises
        ------
        ValueError
            If the conversion is not possible.
        """
        if isinstance(unit, Minutes):
            return value
        
        value = self._pre_convert(value, exp)

        if isinstance(unit, Seconds):
            value = minutes_to_seconds(value)
        elif isinstance(unit, Hours):
            value = minutes_to_hours(value)
        elif isinstance(unit, Days):
            value = minutes_to_days(value)
        else:
            raise ValueError(f"Cannot convert {self} to {unit}")
    
        return self._post_convert(value, exp)


class Hours(Unit):
    """
    A class representing hours as a unit of measurement.

    Attributes
    ----------
    name : str
        The name of the unit (set to "h").
    """

    name: str = eqx.field(static=True, default_factory=lambda: "h")

    def convert_to(self, unit: Unit, value: ArrayLike, exp: int | float = 1, *args) -> ArrayLike:
        """
        Converts the value to the specified unit.

        Parameters
        ----------
        unit : Unit
            The unit to convert to.
        value : ArrayLike
            The value to convert.
        exp : int or float, optional
            The exponent to use for conversion (default is 1).
        *args
            Additional arguments for conversion.

        Returns
        -------
        ArrayLike
            The converted value.

        Raises
        ------
        ValueError
            If the conversion is not possible.
        """
        if isinstance(unit, Hours):
            return value
        
        value = self._pre_convert(value, exp)

        if isinstance(unit, Seconds):
            value = hours_to_seconds(value)
        elif isinstance(unit, Minutes):
            value = hours_to_minutes(value)
        elif isinstance(unit, Days):
            value = hours_to_days(value)
        else:
            raise ValueError(f"Cannot convert {self} to {unit}")
    
        return self._post_convert(value, exp)


class Days(Unit):
    """
    A class representing days as a unit of measurement.

    Attributes
    ----------
    name : str
        The name of the unit (set to "d").
    """

    name: str = eqx.field(static=True, default_factory=lambda: "d")

    def convert_to(self, unit: Unit, value: ArrayLike, exp: int | float = 1, *args) -> ArrayLike:
        """
        Converts the value to the specified unit.

        Parameters
        ----------
        unit : Unit
            The unit to convert to.
        value : ArrayLike
            The value to convert.
        exp : int or float, optional
            The exponent to use for conversion (default is 1).
        *args
            Additional arguments for conversion.

        Returns
        -------
        ArrayLike
            The converted value.

        Raises
        ------
        ValueError
            If the conversion is not possible.
        """
        if isinstance(unit, Days):
            return value
        
        value = self._pre_convert(value, exp)

        if isinstance(unit, Seconds):
            value = days_to_seconds(value)
        elif isinstance(unit, Minutes):
            value = days_to_minutes(value)
        elif isinstance(unit, Hours):
            value = days_to_hours(value)
        else:
            raise ValueError(f"Cannot convert {self} to {unit}")
    
        return self._post_convert(value, exp)
    

UNIT = {
    "": Dimensionless(),
    "m": Meters(), "km": Kilometers(), "°": LatLonDegrees(),
    "s": Seconds(), "min": Minutes(), "h": Hours(), "d": Days()
}
"""
A dictionary mapping unit symbols to their corresponding Unit objects.

Keys
----
"" : Dimensionless
    Represents a dimensionless unit.
"m" : Meters
    Represents meters as a unit of measurement.
"km" : Kilometers
    Represents kilometers as a unit of measurement.
"°" : LatLonDegrees
    Represents latitude and longitude degrees as a unit of measurement.
"s" : Seconds
    Represents seconds as a unit of measurement.
"min" : Minutes
    Represents minutes as a unit of measurement.
"h" : Hours
    Represents hours as a unit of measurement.
"d" : Days
    Represents days as a unit of measurement.

Values
------
Unit
    The corresponding Unit object for each unit symbol.
"""


def units_to_str(unit: Dict[Unit, int | float]) -> str:
    """
    Converts a dictionary of units with their exponents to a string representation.

    Parameters
    ----------
    unit : Dict[Unit, int or float]
        A dictionary of units with their exponents.

    Returns
    -------
    str
        A string representation of the units with their exponents.
    """
    def get_exp_str(exp: int | float) -> str:
        if exp == 1:
            return ""
        else:
            return f"^{{{Fraction(exp).limit_denominator()}}}"
    
    def get_dim_str(dim: Unit, exp: int | float) -> str:
        if exp == 0:
            return ""
        else:
            return f"{dim}{get_exp_str(exp)}"
    
    return " ".join(get_dim_str(dim, exp) for dim, exp in unit.items()).strip()


def compose_units(
        unit1: Dict[Unit, int | float],
        unit2: Dict[Unit, int | float],
        mul: Literal[-1, 1]
    ) -> Dict[Unit, int | float]:
        """
        Compose two unit dictionaries by combining their values, 
        optionally multiplying the second dictionary's values by a factor to account for multiplication or division.

        Parameters
        ----------
        unit1 : Dict[Unit, int | float]
            The first unit dictionary.
        unit2 : Dict[Unit, int | float]
            The second unit dictionary.
        mul : Literal[-1, 1]
            The multiplier for the second unit dictionary's values.
            Should be either 1 in case of multiplication or -1 in case of division.

        Returns
        -------
        Dict[Unit, int | float]
            The composed unit dictionary, or an empty Dict if both input dictionaries are empty.
        """
        if (not unit1) and (not unit2):
            return {}
        if not unit1:
            return unit2
        if not unit2:
            return unit1
        
        unit = unit1.copy()

        for k, v in unit2.items():
            if isinstance(k, Dimensionless):
                continue
            v *= mul
            if k in unit:
                unit[k] += v
            else:
                unit[k] = v
        
        return unit


def meters_to_kilometers(arr: ArrayLike) -> ArrayLike:
    """
    Converts an array of distances from meters to kilometers.

    Parameters
    ----------
    arr : ArrayLike
        An array of distances in meters.

    Returns
    -------
    ArrayLike
        An array of distances in kilometers.
    """
    return arr / 1000


def meters_to_degrees(arr: Float[Array, "... 2"], latitude: Float[Array, "..."]) -> Float[Array, "... 2"]:
    """
    Converts an array of latitude/longitude distances from meters to degrees.

    Parameters
    ----------
    arr : Float[Array, "... 2"]
        An array of latitude/longitude distances in meters.
    latitude : Float[Array, "..."]
        The latitude in degrees at which the conversion is to be performed.

    Returns
    -------
    Float[Array, "... 2"]
        An array of latitude/longitude distances in degrees.

    Notes
    -----
    This function uses the Haversine formula for accurate conversion of distances.
    """
    dy, dx = arr[..., 0], arr[..., 1]
    lat1_rad = jnp.radians(latitude)

    distance = jnp.sqrt(dx**2 + dy**2)
    bearing = jnp.atan2(dx, dy)

    # using Haversine formula
    lat2_rad = jnp.asin(
        jnp.sin(lat1_rad) * jnp.cos(distance / EARTH_RADIUS) + 
        jnp.cos(lat1_rad) * jnp.sin(distance / EARTH_RADIUS) * jnp.cos(bearing)
    )
    
    dlon = jnp.atan2(
        jnp.sin(bearing) * jnp.sin(distance / EARTH_RADIUS) * jnp.cos(lat1_rad),
        jnp.cos(distance / EARTH_RADIUS) - jnp.sin(lat1_rad) * jnp.sin(lat2_rad)
    )
    
    dlat = jnp.degrees(lat2_rad - lat1_rad)
    dlon = jnp.degrees(dlon)

    return jnp.stack([dlat, dlon], axis=-1)


def kilometers_to_meters(arr: ArrayLike) -> ArrayLike:
    """
    Converts an array of distances from kilometers to meters.

    Parameters
    ----------
    arr : ArrayLike
        An array of distances in kilometers.

    Returns
    -------
    ArrayLike
        An array of distances in meters.
    """
    return arr * 1000


def kilometers_to_degrees(arr: Float[Array, "... 2"], latitude: Float[Array, "..."]) -> Float[Array, "... 2"]:
    """
    Converts an array of latitude/longitude distances from kilometers to degrees.

    Parameters
    ----------
    arr : Float[Array, "... 2"]
        An array of latitude/longitude distances in kilometers.
    latitude : Float[Array, "..."]
        The latitude in degrees at which the conversion is to be performed.

    Returns
    -------
    Float[Array, "... 2"]
        An array of latitude/longitude distances in degrees.

    Notes
    -----
    This function uses the Haversine formula for accurate conversion of distances.
    """
    return meters_to_degrees(kilometers_to_meters(arr), latitude)


def degrees_to_meters(arr: Float[Array, "... 2"], latitude: Float[Array, "..."]) -> Float[Array, "... 2"]:
    """
    Converts an array of latitude/longitude distances from degrees to meters.

    Parameters
    ----------
    arr : Float[Array, "... 2"]
        An array of latitude/longitude distances in degrees.
    latitude : Float[Array, "..."]
        The latitude in degrees at which the conversion is to be performed.

    Returns
    -------
    Float[Array, "... 2"]
        An array of latitude/longitude distances in meters.

    Notes
    -----
    This function uses the Haversine formula for accurate conversion of distances.
    """
    dlat, dlon = arr[..., 0], arr[..., 1]
    
    lat1 = latitude
    lat2 = lat1 + dlat

    dy = distance_on_earth(
        jnp.stack([lat1, jnp.zeros_like(lat1)], axis=-1), jnp.stack([lat2, jnp.zeros_like(lat2)], axis=-1)
    )
    dx = distance_on_earth(jnp.stack([lat1, jnp.zeros_like(lat1)], axis=-1), jnp.stack([lat1, dlon], axis=-1))

    return jnp.stack([dy, dx], axis=-1)


def degrees_to_kilometers(arr: Float[Array, "... 2"], latitude: Float[Array, "..."]) -> Float[Array, "... 2"]:
    """
    Converts an array of latitude/longitude distances from degrees to kilometers.

    Parameters
    ----------
    arr : Float[Array, "... 2"]
        An array of latitude/longitude distances in degrees.
    latitude : Float[Array, "..."]
        The latitude in degrees at which the conversion is to be performed.

    Returns
    -------
    Float[Array, "... 2"]
        An array of latitude/longitude distances in kilometers.

    Notes
    -----
    This function uses the Haversine formula for accurate conversion of distances.
    """
    return meters_to_kilometers(degrees_to_meters(arr, latitude))


def seconds_to_minutes(arr: ArrayLike) -> ArrayLike:
    """
    Converts an array of time durations from seconds to minutes.

    Parameters
    ----------
    arr : ArrayLike
        An array of time durations in seconds.

    Returns
    -------
    ArrayLike
        An array of time durations in minutes.
    """
    return arr / 60


def seconds_to_hours(arr: ArrayLike) -> ArrayLike:
    """
    Converts an array of time durations from seconds to hours.

    Parameters
    ----------
    arr : ArrayLike
        An array of time durations in seconds.

    Returns
    -------
    ArrayLike
        An array of time durations in hours.
    """
    return minutes_to_hours(seconds_to_minutes(arr))


def seconds_to_days(arr: ArrayLike) -> ArrayLike:
    """
    Converts an array of time durations from seconds to days.

    Parameters
    ----------
    arr : ArrayLike
        An array of time durations in seconds.

    Returns
    -------
    ArrayLike
        An array of time durations in days.
    """
    return minutes_to_days(seconds_to_minutes(arr))


def minutes_to_seconds(arr: ArrayLike) -> ArrayLike:
    """
    Converts an array of time durations from minutes to seconds.

    Parameters
    ----------
    arr : ArrayLike
        An array of time durations in minutes.

    Returns
    -------
    ArrayLike
        An array of time durations in seconds.
    """
    return arr * 60


def minutes_to_hours(arr: ArrayLike) -> ArrayLike:
    """
    Converts an array of time durations from minutes to hours.

    Parameters
    ----------
    arr : ArrayLike
        An array of time durations in minutes.

    Returns
    -------
    ArrayLike
        An array of time durations in hours.
    """
    return arr / 60


def minutes_to_days(arr: ArrayLike) -> ArrayLike:
    """
    Converts an array of time durations from minutes to days.

    Parameters
    ----------
    arr : ArrayLike
        An array of time durations in minutes.

    Returns
    -------
    ArrayLike
        An array of time durations in days.
    """
    return hours_to_days(minutes_to_hours(arr))


def hours_to_seconds(arr: ArrayLike) -> ArrayLike:
    """
    Converts an array of time durations from hours to seconds.

    Parameters
    ----------
    arr : ArrayLike
        An array of time durations in hours.

    Returns
    -------
    ArrayLike
        An array of time durations in seconds.
    """
    return minutes_to_seconds(hours_to_minutes(arr))


def hours_to_minutes(arr: ArrayLike) -> ArrayLike:
    """
    Converts an array of time durations from hours to minutes.

    Parameters
    ----------
    arr : ArrayLike
        An array of time durations in hours.

    Returns
    -------
    ArrayLike
        An array of time durations in minutes.
    """
    return arr * 60


def hours_to_days(arr: ArrayLike) -> ArrayLike:
    """
    Converts an array of time durations from hours to days.

    Parameters
    ----------
    arr : ArrayLike
        An array of time durations in hours.

    Returns
    -------
    ArrayLike
        An array of time durations in days.
    """
    return arr / 24


def days_to_seconds(arr: ArrayLike) -> ArrayLike:
    """
    Converts an array of time durations from days to seconds.

    Parameters
    ----------
    arr : ArrayLike
        An array of time durations in days.

    Returns
    -------
    ArrayLike
        An array of time durations in seconds.
    """
    return hours_to_seconds(days_to_hours(arr))


def days_to_minutes(arr: ArrayLike) -> ArrayLike:
    """
    Converts an array of time durations from days to minutes.

    Parameters
    ----------
    arr : ArrayLike
        An array of time durations in days.

    Returns
    -------
    ArrayLike
        An array of time durations in minutes.
    """
    return hours_to_minutes(days_to_hours(arr))


def days_to_hours(arr: ArrayLike) -> ArrayLike:
    """
    Converts an array of time durations from days to hours.

    Parameters
    ----------
    arr : ArrayLike
        An array of time durations in days.

    Returns
    -------
    ArrayLike
        An array of time durations in hours.
    """
    return arr * 24


def time_in_seconds(arr: ArrayLike) -> ArrayLike:
    """
    Converts an array of datetime64 values to seconds since the Unix epoch.

    Parameters
    ----------
    arr : ArrayLike
        An array of datetime64 values or a single datetime64 value.

    Returns
    -------
    ArrayLike
        An array of integers representing the number of seconds since the Unix epoch.
    """
    if (isinstance(arr, np.datetime64) or (isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.datetime64))):
        arr = arr.astype("datetime64[s]").astype(int)

    return arr
