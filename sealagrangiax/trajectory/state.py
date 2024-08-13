from __future__ import annotations
from numbers import Number

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ..utils.unit import (  # noqa
    degrees_to_meters, degrees_to_kilometers, degrees_to_radians, kilometers_to_degrees, kilometers_to_meters,
    longitude_in_0_360_degrees, meters_to_degrees, meters_to_kilometers
)
from ..utils import UNIT, WHAT
from ._state import State
from ._utils import earth_distance


class Location(State):
    value: Float[Array, "2"] = eqx.field(converter=longitude_in_0_360_degrees)

    def __init__(self, value: Float[Array, "2"], **_):
        super().__init__(value, what=WHAT.location, unit=UNIT.degrees)

    @property
    def latitude(self) -> Float[Array, ""]:
        return self.value[0]

    @property
    def longitude(self) -> Float[Array, ""]:
        return self.value[1]

    @eqx.filter_jit
    def earth_distance(self, other: Location) -> Float[Array, ""]:
        return earth_distance(self.value, other.value)

    @eqx.filter_jit
    def _check_other_type(self, other: State | Array | Number) -> State | Array | Number:
        if isinstance(other, Displacement):
            if self.unit != other.unit:
                other = other.convert_to(self.unit, self.latitude)

        return super()._check_other_type(other)

    @eqx.filter_jit
    def __add__(self, other: State | Array | Number) -> Float[Array, "state"]:
        other = self._check_other_type(other)

        return self.value + other

    @eqx.filter_jit
    def __sub__(self, other: State | Array | Number) -> Float[Array, "state"]:
        other, kwargs = self._check_other_type(other)

        return self.value - other


class Displacement(State):
    value: Float[Array, "2"]

    def __init__(self, value: Float[Array, "2"], unit: UNIT, **_):
        super().__init__(value, what=WHAT.displacement, unit=unit)

    @property
    def latitude(self) -> Float[Array, ""]:
        return self.value[0]

    @property
    def longitude(self) -> Float[Array, ""]:
        return self.value[1]

    @eqx.filter_jit
    def convert_to(self, unit: UNIT, latitude: Float[Array, ""]) -> Float[Array, "2"]:
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
    value: Int[Array, ""]

    def __init__(self, value: Int[Array, ""]):
        super().__init__(value, what=WHAT.time, unit=UNIT.seconds)
