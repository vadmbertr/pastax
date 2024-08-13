from __future__ import annotations
from numbers import Number
from typing import Callable, ClassVar

import equinox as eqx
from jaxtyping import Array, Float, Int

from ..utils import UNIT
from ._state import WHAT
from ._timeseries import Timeseries


class TimeseriesEnsemble(eqx.Module):
    _members: Timeseries
    _members_type: ClassVar = Timeseries
    size: int

    def __init__(
            self,
            values: Float[Array, "member time state"],
            times: Int[Array, "time"],
            what: WHAT = WHAT.unknown,
            unit: UNIT = UNIT.dimensionless,
            **kwargs
    ):
        self._members = eqx.filter_vmap(
            lambda s: self._members_type(s, times, what=what, unit=unit, **kwargs),
            out_axes=eqx._vmap_pmap.if_mapped(0)  # noqa
        )(values)
        self.size = values.shape[0]

    @property
    def length(self) -> int:
        return self._members.length

    @property
    def states(self) -> Float[Array, "member time state"]:
        return self._members.states

    @property
    def times(self) -> Int[Array, "time"]:
        return self._members.times

    @property
    def unit(self) -> UNIT:
        return self._members.unit

    @property
    def what(self) -> WHAT:
        return self._members.what

    @eqx.filter_jit
    def crps(
            self,
            other: Timeseries,
            distance_func: Callable[[Timeseries, Timeseries], Float[Array, "time"]] = Timeseries.euclidean_distance
    ) -> Float[Array, "time"] | Timeseries:
        biases = self.map(lambda member: distance_func(other, member))
        bias = biases.mean(axis=0)

        n_members = self.size
        dispersion = self.ensemble_dispersion(distance_func)
        dispersion /= n_members * (n_members - 1)

        return bias - dispersion

    @eqx.filter_jit
    def ensemble_dispersion(
            self,
            distance_func: Callable[[Timeseries, Timeseries], Float[Array, "times"]] = Timeseries.euclidean_distance
    ) -> Float[Array, "times"] | Timeseries:
        distances = self.map(lambda member1: self.map(lambda member2: distance_func(member1, member2)))
        # divide by 2 as pairs are duplicated (JAX efficient computation way)
        dispersion = distances.sum((0, 1)) / 2

        return dispersion

    @eqx.filter_jit
    def map(self, func: Callable[[Timeseries], Float[Array, "..."]]) -> Array:
        def axes_func(leaf):
            axes = None
            if eqx.is_array(leaf) and leaf.ndim > 1:
                axes = 0
            return axes

        return eqx.filter_vmap(func, in_axes=axes_func)(self._members)

    @eqx.filter_jit
    def mean(self) -> Float[Array, "time state"]:
        return self.states.mean(axis=0)

    @eqx.filter_jit
    def __sub__(self, other: Timeseries | Float[Array, "time state"] | Number) -> Float[Array, "time state"]:
        return self.map(lambda timeseries: timeseries.__sub__(other))
