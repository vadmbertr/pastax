from __future__ import annotations
from numbers import Number

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from ..utils import UNIT, WHAT


class State(eqx.Module):
    value: Float[Array, "state"]
    what: WHAT
    unit: UNIT

    def __init__(
            self,
            value: Float[Array, "state"],
            what: WHAT = WHAT.unknown,
            unit: UNIT = UNIT.dimensionless
    ):
        self.value = value
        self.what = what
        self.unit = unit

    @eqx.filter_jit
    def euclidean_distance(self, other: State) -> Float[Array, ""]:
        return jnp.sqrt(((self.value - other.value) ** 2).sum())

    @eqx.filter_jit
    def _check_other_type(self, other: State | Array | Number) -> State | Array | Number:
        if isinstance(other, State):
            other = other.value

        return other

    @eqx.filter_jit
    def __add__(self, other: State | Array | Number) -> Float[Array, "state"]:
        other = self._check_other_type(other)

        return self.value + other

    @eqx.filter_jit
    def __mul__(self, other: State | Array | Number) -> Float[Array, "state"]:
        other = self._check_other_type(other)

        return self.value * other

    @eqx.filter_jit
    def __sub__(self, other: State | Array | Number) -> Float[Array, "state"]:
        other = self._check_other_type(other)

        return self.value - other

    @eqx.filter_jit
    def __truediv__(self, other: State | Array | Number) -> Float[Array, "state"]:
        other = self._check_other_type(other)

        return self.value / other

    @eqx.filter_jit
    def __eq__(self, other: State) -> Bool[Array, ""]:
        return self.value.__eq__(other.value).all()

    @eqx.filter_jit
    def __ge__(self, other: State) -> Bool[Array, ""]:
        return self.value.__ge__(other.value).all()

    @eqx.filter_jit
    def __gt__(self, other: State) -> Bool[Array, ""]:
        return self.value.__gt__(other.value).all()

    @eqx.filter_jit
    def __le__(self, other: State) -> Bool[Array, ""]:
        return self.value.__le__(other.value).all()

    @eqx.filter_jit
    def __lt__(self, other: State) -> Bool[Array, ""]:
        return self.value.__lt__(other.value).all()
