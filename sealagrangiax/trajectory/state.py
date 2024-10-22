from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float
import numpy as np

from ..utils.geo import distance_on_earth, longitude_in_180_180_degrees
from ..utils.unit import Unit, UNIT
from ._state import State


def location_converter(x: Float[Array, "... 2"]) -> Float[Array, "... 2"]:
    x = jnp.asarray(x, dtype=float)
    return x.at[..., 1].set(longitude_in_180_180_degrees(x[..., 1]))


class Location(State):
    """
    Class representing a geographical location with latitude and longitude.

    Attributes
    ----------
    _value : Float[Array, "... 2"]
        The latitude and longitude of the location.
    _unit : Dict[Unit, int | float]
        UNIT["°"].
    name : str
        "Location in [latitude, longitude]".

    Methods
    -------
    __init__(value, **_)
        Initializes the Location with given latitude and longitude.
    latitude
        Returns the latitude of the location.
    longitude
        Returns the longitude of the location.
    earth_distance(other)
        Computes the Earth distance between this location and another location.
    """

    _value: Float[Array, "... 2"] = eqx.field(converter=location_converter)

    def __init__(self, value: Float[Array, "... 2"], unit: Unit = UNIT["°"], **_):
        """
        Initializes the Location with given latitude and longitude.

        Parameters
        ----------
        value : Float[Array, "2"]
            The latitude and longitude of the location.
        unit : Unit | Dict[str, Unit], optional
            The unit of the displacement (default is unit.LatLonDegrees()).
        """
        super().__init__(value, unit=unit, name="Location in [latitude, longitude]")

    @property
    def latitude(self) -> State:
        """
        Returns the latitude of the location.

        Returns
        -------
        State
            The latitude of the location.
        """
        return State(self.value[..., 0], unit=UNIT["°"], name="Latitude")

    @property
    def longitude(self) -> State:
        """
        Returns the longitude of the location.

        Returns
        -------
        State
            The longitude of the location.
        """
        return State(self.value[..., 1], unit=UNIT["°"], name="Longitude")

    def distance_on_earth(self, other: Location) -> State:
        """
        Computes the distance in meters between this location and another location.

        Parameters
        ----------
        other : Location
            The other location to compute the distance to.

        Returns
        -------
        State
            The Earth distance in meters between the two locations.

        Notes
        -----
        This function uses the Haversine formula to compute the distance between two points on the Earth surface.
        """
        return State(distance_on_earth(self.value, other.value), unit=UNIT["m"], name="Distance on Earth")


class Displacement(State):
    """
    Class representing a displacement with latitude and longitude components.

    Attributes
    ----------
    _value : Float[Array, "... 2"]
        The latitude and longitude components of the displacement.
    _unit : Dict[Unit, int | float]
        The unit of the displacement.
    name : str
        "Displacement in [latitude, longitude]".

    Methods
    -------
    __init__(value, unit)
        Initializes the Displacement with given latitude and longitude components and unit.
    latitude
        Returns the latitude component of the displacement.
    longitude
        Returns the longitude component of the displacement.
    """

    def __init__(self, value: Float[Array, "... 2"], unit: Unit = UNIT["°"], **_):
        """
        Initializes the Displacement with given latitude and longitude components and unit.

        Parameters
        ----------
        value : Float[Array, "2"]
            The latitude and longitude components of the displacement.
        unit : Unit | Dict[str, Unit], optional
            The unit of the displacement (default is unit.LatLonDegrees()).
        """
        super().__init__(value, unit=unit, name="Displacement in [latitude, longitude]")

    @property
    def latitude(self) -> State:
        """
        Returns the latitude component of the displacement.

        Returns
        -------
        State
            The latitude component of the displacement.
        """
        return State(self.value[..., 0], unit=self.unit, name="Displacement in latitude")
    
    @property
    def longitude(self) -> State:
        """
        Returns the longitude component of the displacement.

        Returns
        -------
        State
            The longitude component of the displacement.
        """
        return State(self.value[..., 1], unit=self.unit, name="Displacement in longitude")


class Time(State):
    """
    Class representing a time value.

    Attributes
    ----------
    _value : ArrayLike
        The time value.
    _unit : Dict[Unit, int | float]
        The unit of the time (default is unit.Seconds()).
    name : str
        "Time since epoch".

    Methods
    -------
    __init__(value, unit)
        Initializes the Time with given time value.
    to_datetime()
        Converts the time value to a numpy array of datetime64[s].
    """

    def __init__(self, value: ArrayLike, unit: Unit = UNIT["s"], **_):
        """
        Initializes the Time with given time value.

        Parameters
        ----------
        value : ArrayLike
            The time value.
        _unit : Unit | Dict[Unit, int | float], optional
            The unit of the time (default is unit.Seconds()).
        """
        super().__init__(value, unit=unit, name="Time since epoch")

    def to_datetime(self) -> np.ndarray:
        """
        Converts the time value to a numpy array of datetime64[s].

        Returns
        -------
        np.ndarray
            The time value as a numpy array of datetime64[s].
        """
        return np.asarray(self.value).astype("datetime64[s]")
