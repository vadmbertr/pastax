from __future__ import annotations
from fractions import Fraction
import functools as ft
from typing import Literal, Dict

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float    
import numpy as np

from .geo import EARTH_RADIUS


ft.total_ordering
class Unit(eqx.Module):
    """
    Base class representing [`pastax.utils.Unit`] of measurement.

    Attributes
    ----------
    name : str
        The name of the [`pastax.utils.Unit`].

    Methods
    -------
    __eq__(other)
        Checks if two [`pastax.utils.Unit`] are equal.
    __lt__(other)
        Checks if one [`pastax.utils.Unit`] is less than another (using their name).
    __hash__()
        Returns the hash of the [`pastax.utils.Unit`].
    __repr__()
        Returns the string representation of the [`pastax.utils.Unit`].
    convert_to(unit, value, exp, *args)
        Converts the `value` to the specified [`pastax.utils.Unit`].
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
        Prepares the value for conversion between "base" [`pastax.utils.Unit`] by raising it to the power of the reciprocal of the 
        exponent.

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
        Finalizes the conversion between "base" [`pastax.utils.Unit`] by raising the value to the power of the exponent.

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
        Converts the `value` to the specified [`pastax.utils.Unit`].

        Parameters
        ----------
        unit : Unit
            The [`pastax.utils.Unit`] to convert to.
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
    

class Meters(Unit):
    """
    Class representing meters as a [`pastax.utils.Unit`] of measurement.

    Attributes
    ----------
    name : str
        The name of the [`pastax.utils.Unit`], set to `"m"`.
    """

    name: str = eqx.field(static=True, default_factory=lambda: "m")

    def convert_to(self, unit: Unit, value: ArrayLike, exp: int | float = 1, *args) -> ArrayLike:
        """
        Converts the value to the specified [`pastax.utils.Unit`].

        Parameters
        ----------
        unit : Unit
            The [`pastax.utils.Unit`] to convert to.
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
    Class representing kilometers as a [`pastax.utils.Unit`] of measurement.

    Attributes
    ----------
    name : str
        The name of the [`pastax.utils.Unit`], set to `"km"`.
    """

    name: str = eqx.field(static=True, default_factory=lambda: "km")

    def convert_to(self, unit: Unit, value: ArrayLike, exp: int | float = 1, *args) -> ArrayLike:
        """
        Converts the value to the specified [`pastax.utils.Unit`].

        Parameters
        ----------
        unit : Unit
            The [`pastax.utils.Unit`] to convert to.
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
    Class representing latitude and longitude degrees as a [`pastax.utils.Unit`] of measurement.

    Attributes
    ----------
    name : str
        The name of the [`pastax.utils.Unit`], set to `"°"`.
    """

    name: str = eqx.field(static=True, default_factory=lambda: "°")

    def convert_to(self, unit: Unit, value: ArrayLike, exp: int | float = 1, *args) -> ArrayLike:
        """
        Converts the value to the specified [`pastax.utils.Unit`].

        Parameters
        ----------
        unit : Unit
            The [`pastax.utils.Unit`] to convert to.
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
    Class representing seconds as a [`pastax.utils.Unit`] of measurement.

    Attributes
    ----------
    name : str
        The name of the [`pastax.utils.Unit`], set to `"s"`.
    """

    name: str = eqx.field(static=True, default_factory=lambda: "s")

    def convert_to(self, unit: Unit, value: ArrayLike, exp: int | float = 1, *args) -> ArrayLike:
        """
        Converts the value to the specified unit.

        Parameters
        ----------
        unit : Unit
            The [`pastax.utils.Unit`] to convert to.
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
    Class representing minutes as a [`pastax.utils.Unit`] of measurement.

    Attributes
    ----------
    name : str
        The name of the [`pastax.utils.Unit`], set to `"min"`.
    """

    name: str = eqx.field(static=True, default_factory=lambda: "min")

    def convert_to(self, unit: Unit, value: ArrayLike, exp: int | float = 1, *args) -> ArrayLike:
        """
        Converts the value to the specified unit.

        Parameters
        ----------
        unit : Unit
            The [`pastax.utils.Unit`] to convert to.
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
    Class representing hours as a [`pastax.utils.Unit`] of measurement.

    Attributes
    ----------
    name : str
        The name of the [`pastax.utils.Unit`], set to `"h"`.
    """

    name: str = eqx.field(static=True, default_factory=lambda: "h")

    def convert_to(self, unit: Unit, value: ArrayLike, exp: int | float = 1, *args) -> ArrayLike:
        """
        Converts the value to the specified unit.

        Parameters
        ----------
        unit : Unit
            The [`pastax.utils.Unit`] to convert to.
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
    Class representing days as a [`pastax.utils.Unit`] of measurement.

    Attributes
    ----------
    name : str
        The name of the [`pastax.utils.Unit`], set to `"d"`.
    """

    name: str = eqx.field(static=True, default_factory=lambda: "d")

    def convert_to(self, unit: Unit, value: ArrayLike, exp: int | float = 1, *args) -> ArrayLike:
        """
        Converts the value to the specified unit.

        Parameters
        ----------
        unit : Unit
            The [`pastax.utils.Unit`] to convert to.
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
    "m": Meters(), "km": Kilometers(), "°": LatLonDegrees(),
    "s": Seconds(), "min": Minutes(), "h": Hours(), "d": Days()
}
"""
A dictionary mapping unit symbols to their corresponding [`pastax.utils.Unit`] objects.

Keys
----
"m" : Meters
    Represents meters as a [`pastax.utils.Unit`] of measurement.
"km" : Kilometers
    Represents kilometers as a [`pastax.utils.Unit`] of measurement.
"°" : LatLonDegrees
    Represents latitude and longitude degrees as a [`pastax.utils.Unit`] of measurement.
"s" : Seconds
    Represents seconds as a [`pastax.utils.Unit`] of measurement.
"min" : Minutes
    Represents minutes as a [`pastax.utils.Unit`] of measurement.
"h" : Hours
    Represents hours as a [`pastax.utils.Unit`] of measurement.
"d" : Days
    Represents days as a [`pastax.utils.Unit`] of measurement.

Values
------
Unit
    The corresponding [`pastax.utils.Unit`] object for each unit symbol.
"""


def units_to_str(unit: Dict[Unit, int | float]) -> str:
    """
    Converts a dictionary of [`pastax.utils.Unit`] with their exponents to a string representation.

    Parameters
    ----------
    unit : Dict[Unit, int or float]
        A dictionary of [`pastax.utils.Unit`] with their exponents.

    Returns
    -------
    str
        A string representation of the [`pastax.utils.Unit`] with their exponents.
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
        Compose two [`pastax.utils.Unit`] dictionaries by combining their values, 
        optionally multiplying the second dictionary's values by a factor to account for multiplication or division.

        Parameters
        ----------
        unit1 : Dict[Unit, int | float]
            The first [`pastax.utils.Unit`] dictionary.
        unit2 : Dict[Unit, int | float]
            The second [`pastax.utils.Unit`] dictionary.
        mul : Literal[-1, 1]
            The multiplier for the second [`pastax.utils.Unit`] dictionary's values.
            Should be either 1 in case of multiplication or -1 in case of division.

        Returns
        -------
        Dict[Unit, int | float]
            The composed [`pastax.utils.Unit`] dictionary, or an empty dictionary if both input dictionaries are empty.
        """
        if (not unit1) and (not unit2):
            return {}
        if not unit1:
            return unit2
        if not unit2:
            return unit1
        
        unit = unit1.copy()

        for k, v in unit2.items():
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
    """
    arr = jnp.degrees(arr / EARTH_RADIUS)
    arr = arr.at[..., 1].divide(jnp.cos(jnp.radians(latitude)))
    return arr


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
    """
    arr = jnp.radians(arr) * EARTH_RADIUS
    arr = arr.at[..., 1].multiply(jnp.cos(jnp.radians(latitude)))
    return arr


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
