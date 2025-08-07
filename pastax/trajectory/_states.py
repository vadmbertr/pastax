from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Real

from ..utils._geo import distance_on_earth, longitude_in_180_180_degrees
from ..utils._unit import UNIT, Unit
from ._state import State
from ._unitful import unit_converter


class Location(State):
    """
    Class representing a geographical location with latitude and longitude.

    Attributes
    ----------
    name : str
        The name of the [`pastax.trajectory.Location`][], set to `"Location in [latitude, longitude]"`.

    Methods
    -------
    __init__(value, **_)
        Initializes the [`pastax.trajectory.Location`][] with given latitude and longitude.
    latitude
        Returns the latitude of the [`pastax.trajectory.Location`][].
    longitude
        Returns the longitude of the [`pastax.trajectory.Location`][].
    distance_on_earth(other)
        Computes the Earth distance between this [`pastax.trajectory.Location`][] and another
        [`pastax.trajectory.Location`][].
    """

    _value: Float[Array, "... 2"] = eqx.field(converter=lambda x: jnp.asarray(x, dtype=float))

    def __init__(
        self,
        value: Float[Any, "... 2"],
        unit: dict[Unit, int | float] = unit_converter(UNIT["°"]),
        **_,
    ):
        """
        Initializes the [`pastax.trajectory.Location`][] with given latitude and longitude.

        Parameters
        ----------
        value : Float[Array, "... 2"] | Sequence[float]
            The latitude and longitude of the location.
        unit : dict[str, Unit], optional
            The [`pastax.utils.Unit`][] of the location, defaults to [`pastax.utils.LatLonDegrees`][].
        """
        if unit == unit_converter(UNIT["°"]):
            value = jnp.asarray(value)
            value = value.at[..., 1].set(longitude_in_180_180_degrees(value[..., 1]))

        super().__init__(value, unit=unit, name="Location in [latitude, longitude]")

    @property
    def latitude(self) -> State:
        """
        Returns the latitude of the [`pastax.trajectory.Location`][].

        Returns
        -------
        State
            The latitude of the [`pastax.trajectory.Location`][].
        """
        return State(self.value[..., 0], unit=UNIT["°"], name="Latitude")

    @property
    def longitude(self) -> State:
        """
        Returns the longitude of the [`pastax.trajectory.Location`][].

        Returns
        -------
        State
            The longitude of the [`pastax.trajectory.Location`][].
        """
        return State(self.value[..., 1], unit=UNIT["°"], name="Longitude")

    def distance_on_earth(self, other: Location) -> State:
        """
        Computes the distance in meters between this [`pastax.trajectory.Location`][]
        and another [`pastax.trajectory.Location`][].

        Parameters
        ----------
        other : Location
            The other [`pastax.trajectory.Location`][] to compute the distance to.

        Returns
        -------
        State
            The Earth distance in meters between the two [`pastax.trajectory.Location`][].

        Notes
        -----
        This function uses the Haversine formula to compute the distance between two points on the Earth surface.
        """
        if not self.unit == unit_converter(UNIT["°"]) or not other.unit == unit_converter(UNIT["°"]):
            raise ValueError("Both locations must be in degrees.")

        return State(
            distance_on_earth(self.value, other.value),
            unit=UNIT["m"],
            name="Distance on Earth",
        )


class Displacement(State):
    """
    Class representing a [`pastax.trajectory.Displacement`][] with latitude and longitude components.

    Attributes
    ----------
    name : str
        The name of the [`pastax.trajectory.Displacement`], set to `"Displacement in [latitude, longitude]"`.

    Methods
    -------
    __init__(value, unit)
        Initializes the [`pastax.trajectory.Displacement`][] with given latitude and longitude components and unit.
    latitude
        Returns the latitude component of the [`pastax.trajectory.Displacement`].
    longitude
        Returns the longitude component of the [`pastax.trajectory.Displacement`].
    """

    def __init__(
        self,
        value: Real[Any, "... 2"],
        unit: dict[Unit, int | float] = UNIT["°"],
        **_,
    ):
        """
        Initializes the [`pastax.trajectory.Displacement`][] with given latitude and longitude components and
        [`pastax.utils.Unit`][].

        Parameters
        ----------
        value : Real[Any, "2"]
            The latitude and longitude components of the [`pastax.trajectory.Displacement`].
        unit : dict[str, Unit], optional
            The [`pastax.utils.Unit`][] of the [`pastax.trajectory.Displacement`], defaults to
            [`pastax.utils.LatLonDegrees`].
        """
        super().__init__(value, unit=unit, name="Displacement in [latitude, longitude]")

    @property
    def latitude(self) -> State:
        """
        Returns the latitude component of the [`pastax.trajectory.Displacement`].

        Returns
        -------
        State
            The latitude component of the [`pastax.trajectory.Displacement`].
        """
        return State(self.value[..., 0], unit=self.unit, name="Displacement in latitude")

    @property
    def longitude(self) -> State:
        """
        Returns the longitude component of the [`pastax.trajectory.Displacement`].

        Returns
        -------
        State
            The longitude component of the [`pastax.trajectory.Displacement`].
        """
        return State(self.value[..., 1], unit=self.unit, name="Displacement in longitude")


class Time(State):
    """
    Class representing a [`pastax.trajectory.Time`][] value.

    Attributes
    ----------
    name : str
        The name of the [`pastax.trajectory.Time`], set to `"Time since epoch"`.

    Methods
    -------
    __init__(value, unit)
        Initializes the [`pastax.trajectory.Time`][] with given time value.
    to_datetime()
        Converts the [`pastax.trajectory.Time`][] to a `numpy.ndarray` of `datetime64[s]`.
    """

    def __init__(self, value: Real[Any, "..."], unit: dict[Unit, int | float] = UNIT["s"], **_):
        """
        Initializes the [`pastax.trajectory.Time`][] with given time value.

        Parameters
        ----------
        value : Real[Any, "..."]
            The time value.
        unit : dict[Unit, int | float], optional
            The [`pastax.utils.Unit`][] of the [`pastax.trajectory.Time`], defaults to [`pastax.utils.Seconds`][].
        """
        super().__init__(value, unit=unit, name="Time since epoch")

    def to_datetime(self) -> np.ndarray:
        """
        Converts the [`pastax.trajectory.Time`][] to a `numpy.ndarray` of `datetime64[s]`.

        Returns
        -------
        np.ndarray
            The [`pastax.trajectory.Time`][] as a `numpy.ndarray` of `datetime64[s]`.
        """
        return np.asarray(self.value).astype("datetime64[s]")
