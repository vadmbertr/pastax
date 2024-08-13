from __future__ import annotations
from numbers import Number
from typing import Callable, ClassVar

import equinox as eqx
from jaxtyping import Array, Float, Int

from ..utils import UNIT
from ._state import State, WHAT
from .state import Time


class Timeseries(eqx.Module):
    _states: State
    _states_type: ClassVar = State
    _times: Time
    length: int

    def __init__(
            self,
            values: Float[Array, "time state"],
            times: Int[Array, "time"],
            what: WHAT = WHAT.unknown,
            unit: UNIT = UNIT.dimensionless,
            **_
    ):
        assert values.shape[0] == times.shape[0], f"values {values} and times {times} have incompatible shapes"

        self._states = eqx.filter_vmap(
            lambda value: self._states_type(value, what=what, unit=unit),
            out_axes=eqx._vmap_pmap.if_mapped(0)  # noqa
        )(values)
        self._times = eqx.filter_vmap(
            lambda time: Time(time),
            out_axes=eqx._vmap_pmap.if_mapped(0)  # noqa
        )(times)
        self.length = times.size

    @property
    def states(self) -> Float[Array, "time state"]:
        return self._states.value

    @property
    def times(self) -> Int[Array, "time"]:
        return self._times.value

    @property
    def unit(self) -> UNIT:
        return self._states.unit

    @property
    def what(self) -> WHAT:
        return self._states.what

    @eqx.filter_jit
    def euclidean_distance(self, other: Timeseries) -> Float[Array, "time"]:
        def axes_func(leaf):
            axes = None
            if eqx.is_array(leaf) and leaf.ndim > 0:
                axes = 0
            return axes

        return eqx.filter_vmap(lambda p1, p2: p1.euclidean_distance(p2), in_axes=axes_func)(self._states, other._states)

    @eqx.filter_jit
    def map(self, func: Callable[[State], Float[Array, "..."]]) -> Float[Array, "time ..."]:
        def axes_func(leaf):
            axes = None
            if eqx.is_array(leaf) and leaf.ndim > 0:
                axes = 0
            return axes

        return eqx.filter_vmap(func, in_axes=axes_func)(self._states)

    @eqx.filter_jit
    def __sub__(
            self,
            other: "TimeseriesEnsembleBase" | Timeseries | Float[Array, "time state"] | Float[Array, "state"] | Float[Array, ""] | Number  # noqa
    ) -> Float[Array, "time state"]:
        from ._ensemble import TimeseriesEnsemble
        if isinstance(other, TimeseriesEnsemble):
            return other.map(lambda other_ts: self.__sub__(other_ts))

        if isinstance(other, Timeseries):
            other = other._states

        return self._states.__sub__(other)
