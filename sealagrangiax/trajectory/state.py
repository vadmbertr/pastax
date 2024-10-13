from __future__ import annotations
from numbers import Number

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import numpy as np

from ..utils.unit import (
    degrees_to_meters, degrees_to_kilometers, kilometers_to_degrees, kilometers_to_meters,
    longitude_in_0_360_degrees, meters_to_degrees, meters_to_kilometers
)
from ..utils import UNIT, WHAT
from ..utils.geo import earth_distance
from ._state import State


@jax.jit
def location_converter(latlon: Float[Array, "2"]):
    return latlon.at[..., 1].set(longitude_in_0_360_degrees(latlon[..., 1]))


class Location(State):
    """
    Class representing a geographical location with latitude and longitude.

    Attributes
    ----------
    value : Float[Array, "2"]
        The latitude and longitude of the location.

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
    _get_other_numeric(other)
        Converts the other operand to a numeric type if it is a Displacement instance.
    __add__(other)
        Adds another state, array, or number to this location.
    __sub__(other)
        Subtracts another state, array, or number from this location.
    """

    value: Float[Array, "2"] = eqx.field(converter=location_converter)

    def __init__(self, value: Float[Array, "2"], **_):
        """
        Initializes the Location with given latitude and longitude.

        Parameters
        ----------
        value : Float[Array, "2"]
            The latitude and longitude of the location.
        **_
            Additional keyword arguments.
        """
        super().__init__(value, what=WHAT.location, unit=UNIT.degrees)

    @property
    def latitude(self) -> Float[Array, ""]:
        """
        Returns the latitude of the location.

        Returns
        -------
        Float[Array, ""]
            The latitude of the location.
        """
        return self.value[..., 0]

    @property
    def longitude(self) -> Float[Array, ""]:
        """
        Returns the longitude of the location.

        Returns
        -------
        Float[Array, ""]
            The longitude of the location.
        """
        return self.value[..., 1]

    @eqx.filter_jit
    def earth_distance(self, other: Location) -> Float[Array, ""]:
        """
        Computes the Earth distance between this location and another location.

        Parameters
        ----------
        other : Location
            The other location to compute the distance to.

        Returns
        -------
        Float[Array, ""]
            The Earth distance between the two locations.
        """
        return earth_distance(self.value, other.value)

    @eqx.filter_jit
    def _get_other_numeric(self, other: State | Array | Number) -> State | Array | Number:
        """
        Converts the other operand to a numeric type if it is a Displacement instance.

        Parameters
        ----------
        other : State or Array or Number
            The other operand to check and convert.

        Returns
        -------
        State or Array or Number
            The converted operand.

        Notes
        -----
        If the other operand is a Displacement instance, it is converted to the same unit as the Location.
        """
        if isinstance(other, Displacement):
            other = other.convert_to(self.unit, self.latitude)

        return super()._get_other_numeric(other)

    @eqx.filter_jit
    def __add__(self, other: State | Array | Number) -> Float[Array, "state"]:
        """
        Adds another state, array, or number to this location.

        Parameters
        ----------
        other : State or Array or Number
            The other operand to add.

        Returns
        -------
        Float[Array, "state"]
            The result of the addition.
        """
        other = self._get_other_numeric(other)

        return self.value + other

    @eqx.filter_jit
    def __sub__(self, other: State | Array | Number) -> Float[Array, "state"]:
        """
        Subtracts another state, array, or number from this location.

        Parameters
        ----------
        other : State or Array or Number
            The other operand to subtract.

        Returns
        -------
        Float[Array, "state"]
            The result of the subtraction.
        """
        other = self._get_other_numeric(other)

        return self.value - other


class Displacement(State):
    """
    Class representing a displacement with latitude and longitude components.

    Attributes
    ----------
    value : Float[Array, "2"]
        The latitude and longitude components of the displacement.

    Methods
    -------
    __init__(value, unit, **_)
        Initializes the Displacement with given latitude and longitude components and unit.
    latitude
        Returns the latitude component of the displacement.
    longitude
        Returns the longitude component of the displacement.
    convert_to(unit, latitude)
        Converts the displacement to the specified unit.
    _to_degrees(latitude)
        Converts the displacement to degrees.
    _to_meters(latitude)
        Converts the displacement to meters.
    _to_kilometers(latitude)
        Converts the displacement to kilometers.
    """

    value: Float[Array, "2"]

    def __init__(self, value: Float[Array, "2"], unit: UNIT, **_):
        """
        Initializes the Displacement with given latitude and longitude components and unit.

        Parameters
        ----------
        value : Float[Array, "2"]
            The latitude and longitude components of the displacement.
        unit : UNIT
            The unit of the displacement.
        **_
            Additional keyword arguments.
        """
        super().__init__(value, what=WHAT.displacement, unit=unit)

    @property
    def latitude(self) -> Float[Array, ""]:
        """
        Returns the latitude component of the displacement.

        Returns
        -------
        Float[Array, ""]
            The latitude component of the displacement.
        """
        return self.value[..., 0]

    @property
    def longitude(self) -> Float[Array, ""]:
        """
        Returns the longitude component of the displacement.

        Returns
        -------
        Float[Array, ""]
            The longitude component of the displacement.
        """
        return self.value[..., 1]

    @eqx.filter_jit
    def convert_to(self, unit: UNIT, latitude: Float[Array, ""]) -> Float[Array, "2"]:
        """
        Converts the displacement to the specified unit.

        Parameters
        ----------
        unit : UNIT
            The unit to convert to.
        latitude : Float[Array, ""]
            The latitude to use for accurate conversion.

        Returns
        -------
        Float[Array, "2"]
            The converted displacement.

        Notes
        -----
        Conversions use the Haversine formula for accuracy.
        """
        value = jax.lax.cond(
            unit == UNIT.degrees,
            lambda: self._to_degrees(latitude),
            lambda: jax.lax.cond(
                unit == UNIT.meters,
                lambda: self._to_meters(latitude),
                lambda: jax.lax.cond(
                    unit == UNIT.kilometers,
                    lambda: self._to_kilometers(latitude),
                    lambda: jnp.full_like(self.value, jnp.nan)
                )
            )
        )

        return value

    @eqx.filter_jit
    def _to_degrees(self, latitude: Float[Array, ""]) -> Float[Array, "2"]:
        """
        Converts the displacement to degrees.

        Parameters
        ----------
        latitude : Float[Array, ""]
            The latitude to use for conversion.

        Returns
        -------
        Float[Array, "2"]
            The displacement in degrees.
        """
        value = jax.lax.cond(
            self.unit == UNIT.degrees,
            lambda: self.value,
            lambda: jax.lax.cond(
                self.unit == UNIT.meters,
                lambda: meters_to_degrees(self.value, latitude),
                lambda: jax.lax.cond(
                    self.unit == UNIT.kilometers,
                    lambda: kilometers_to_degrees(self.value, latitude),
                    lambda: jnp.full_like(self.value, jnp.nan)
                )
            )
        )

        return value

    @eqx.filter_jit
    def _to_meters(self, latitude: Float[Array, ""]) -> Float[Array, "2"]:
        """
        Converts the displacement to meters.

        Parameters
        ----------
        latitude : Float[Array, ""]
            The latitude to use for conversion.

        Returns
        -------
        Float[Array, "2"]
            The displacement in meters.
        """
        value = jax.lax.cond(
            self.unit == UNIT.meters,
            lambda: self.value,
            lambda: jax.lax.cond(
                self.unit == UNIT.degrees,
                lambda: degrees_to_meters(self.value, latitude),
                lambda: jax.lax.cond(
                    self.unit == UNIT.kilometers,
                    lambda: kilometers_to_meters(self.value),
                    lambda: jnp.full_like(self.value, jnp.nan)
                )
            )
        )

        return value

    @eqx.filter_jit
    def _to_kilometers(self, latitude: Float[Array, ""]) -> Float[Array, "2"]:
        """
        Converts the displacement to kilometers.

        Parameters
        ----------
        latitude : Float[Array, ""]
            The latitude to use for conversion.

        Returns
        -------
        Float[Array, "2"]
            The displacement in kilometers.
        """
        value = jax.lax.cond(
            self.unit == UNIT.kilometers,
            lambda: self.value,
            lambda: jax.lax.cond(
                self.unit == UNIT.degrees,
                lambda: degrees_to_kilometers(self.value, latitude),
                lambda: jax.lax.cond(
                    self.unit == UNIT.meters,
                    lambda: meters_to_kilometers(self.value),
                    lambda: jnp.full_like(self.value, jnp.nan)
                )
            )
        )

        return value


class Time(State):
    """
    Class representing a time value.

    Attributes
    ----------
    value : Int[Array, ""]
        The time value.

    Methods
    -------
    __init__(value)
        Initializes the Time with given time value.
    """
    value: Int[Array, ""]

    def __init__(self, value: Int[Array, ""]):
        """
        Initializes the Time with given time value.

        Parameters
        ----------
        value : Int[Array, ""]
            The time value.
        """
        super().__init__(value, what=WHAT.time, unit=UNIT.seconds)

    def to_datetime(self):
        """
        Converts the time value to a numpy array of datetime64[s].

        Returns
        -------
        np.ndarray
            The time value as a numpy array of datetime64[s].
        """
        return np.asarray(self.value).astype("datetime64[s]")
