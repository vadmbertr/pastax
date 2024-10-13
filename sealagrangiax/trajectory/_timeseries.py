from __future__ import annotations
from numbers import Number
from typing import Callable, ClassVar

import equinox as eqx
from jaxtyping import Array, Float, Int
import xarray as xr

from ..utils import UNIT
from ._state import State, WHAT
from .state import Time


class Timeseries(eqx.Module):
    """
    Base class representing a timeseries of states.

    Attributes
    ----------
    _states : State
        The states of the timeseries.
    _states_type : ClassVar
        The type of the states in the timeseries.
    _times : Time
        The time points of the timeseries.
    length : int
        The length of the timeseries.

    Methods
    -------
    __init__(values, times, what=WHAT.unknown, unit=UNIT.dimensionless, **_)
        Initializes the Timeseries with given values, times, and optional parameters.
    states
        Returns the states of the timeseries.
    times
        Returns the time points of the timeseries.
    unit
        Returns the unit of the timeseries.
    what
        Returns the type of the timeseries.
    euclidean_distance(other)
        Computes the Euclidean distance between this timeseries and another timeseries.
    map(func)
        Applies a function to each state in the timeseries.
    __sub__(other)
        Subtracts another timeseries or value from this timeseries.
    """

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
        """
        Initializes the Timeseries with given values, times, and optional parameters.

        Parameters
        ----------
        values : Float[Array, "time state"]
            The values for the states of the timeseries.
        times : Int[Array, "time"]
            The time points for the timeseries.
        what : WHAT, optional
            The type of the timeseries (default is WHAT.unknown).
        unit : UNIT, optional
            The unit of the timeseries (default is UNIT.dimensionless).
        **_
            Additional keyword arguments.
        """
        assert values.ndim > 1, f"values {values} should have 2 dimensions: time and state"
        assert values.shape[0] == times.shape[0], f"values {values} and times {times} have incompatible shapes"

        self._states = eqx.filter_vmap(
            lambda value: self._states_type(value, what=what, unit=unit),
            out_axes=eqx._vmap_pmap.if_mapped(0)
        )(values)
        self._times = eqx.filter_vmap(
            lambda time: Time(time),
            out_axes=eqx._vmap_pmap.if_mapped(0)
        )(times)
        self.length = times.size

    @property
    def states(self) -> Float[Array, "time state"]:
        """
        Returns the states of the timeseries.

        Returns
        -------
        Float[Array, "time state"]
            The states of the timeseries.
        """
        return self._states.value

    @property
    def times(self) -> Int[Array, "time"]:
        """
        Returns the time points of the timeseries.

        Returns
        -------
        Int[Array, "time"]
            The time points of the timeseries.
        """
        return self._times.value

    @property
    def unit(self) -> UNIT:
        """
        Returns the unit of the timeseries.

        Returns
        -------
        UNIT
            The unit of the timeseries.
        """
        return self._states.unit

    @property
    def what(self) -> WHAT:
        """
        Returns the type of the timeseries.

        Returns
        -------
        WHAT
            The type of the timeseries.
        """
        return self._states.what

    @eqx.filter_jit
    def euclidean_distance(self, other: Timeseries) -> Float[Array, "time"]:
        """
        Computes the Euclidean distance between this timeseries and another timeseries.

        Parameters
        ----------
        other : Timeseries
            The other timeseries to compute the distance to.

        Returns
        -------
        Float[Array, "time"]
            The Euclidean distance between the two timeseries.
        """
        def axes_func(leaf):
            axes = None
            if eqx.is_array(leaf) and leaf.ndim > 0:
                axes = 0
            return axes

        return eqx.filter_vmap(lambda p1, p2: p1.euclidean_distance(p2), in_axes=axes_func)(self._states, other._states)

    @eqx.filter_jit
    def map(self, func: Callable[[State], Float[Array, "..."]]) -> Float[Array, "time ..."]:
        """
        Applies a function to each state in the timeseries.

        Parameters
        ----------
        func : Callable[[State], Float[Array, "..."]]
            The function to apply to each state.

        Returns
        -------
        Float[Array, "time ..."]
            The result of applying the function to each state.
        """
        def axes_func(leaf):
            axes = None
            if eqx.is_array(leaf) and leaf.ndim > 0:
                axes = 0
            return axes

        return eqx.filter_vmap(func, in_axes=axes_func)(self._states)

    def to_dataarray(self) -> xr.DataArray:
        """
        Converts the timeseries states to an xarray DataArray.

        Returns
        -------
        xr.DataArray
            An xarray DataArray containing the timeseries states.
        """
        da = xr.DataArray(
            data=self.states,
            dims=["time"],
            coords={"time": self._times.to_datetime()},
            name=WHAT[self.what],
            attrs={"units": UNIT[self.unit]}
        )

        return da

    def to_dataset(self) -> xr.Dataset:
        """
        Converts the timeseries states to an xarray Dataset.

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the timeseries states.
        """
        da = self.to_dataarray()
        ds = da.to_dataset()

        return ds

    @eqx.filter_jit
    def __sub__(
        self,
        other: "TimeseriesEnsemble" | Timeseries | Float[Array, "time state"] | Float[Array, "state"] | Float[Array, ""] | Number
    ) -> Float[Array, "time state"]:
        """
        Subtracts another timeseries or value from this timeseries.

        Parameters
        ----------
        other : TimeseriesEnsemble or Timeseries or Float[Array, "time state"] or Float[Array, "state"] or Float[Array, ""] or Number
            The other operand to subtract.

        Returns
        -------
        Float[Array, "time state"]
            The result of the subtraction.
        """
        from ._ensemble import TimeseriesEnsemble
        if isinstance(other, TimeseriesEnsemble):
            return other.map(lambda other_ts: self.__sub__(other_ts))

        if isinstance(other, Timeseries):
            other = other._states

        return self._states.__sub__(other)
