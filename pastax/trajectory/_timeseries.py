from __future__ import annotations
from typing import Callable, ClassVar, Dict

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float
import xarray as xr

from ..utils._unit import Unit, units_to_str
from ._state import State
from ._states import Time
from ._unitful import Unitful


def _in_axes_func(leaf):
    axes = None
    if eqx.is_array(leaf) and leaf.ndim > 0:
        axes = 0
    return axes


class Timeseries(Unitful):
    """
    Class representing a [`pastax.trajectory.Timeseries`].

    Attributes
    ----------
    states : State
        The [`pastax.trajectory.State`][] of the [`pastax.trajectory.Timeseries`].
    _states_type : ClassVar
        The type of the [`pastax.trajectory.State`][] in the [`pastax.trajectory.Timeseries`].
    times : Time
        The [`pastax.trajectory.Time`][] points of the [`pastax.trajectory.Timeseries`].
    length : int
        The length of the [`pastax.trajectory.Timeseries`].

    Methods
    -------
    __init__(states, times, **__)
        Initializes the [`pastax.trajectory.Timeseries`][] with given [`pastax.trajectory.State`][], [`pastax.trajectory.Time`][], and optional parameters.
    value
        Returns the value of the [`pastax.trajectory.Timeseries`].
    unit
        Returns the unit of the [`pastax.trajectory.Timeseries`].
    name
        Returns the name of the [`pastax.trajectory.Timeseries`].
    attach_name(name)
        Attaches a name to the [`pastax.trajectory.Timeseries`].
    euclidean_distance(other)
        Computes the Euclidean distance between this [`pastax.trajectory.Timeseries`][] and another [`pastax.trajectory.Timeseries`].
    map(func)
        Applies a function to each [`pastax.trajectory.State`][] in the [`pastax.trajectory.Timeseries`].
    to_xarray()
        Converts the [`pastax.trajectory.Timeseries`][] to a `xarray.Dataset`.
    from_array(values, times, unit={}, name=None, **kwargs)
        Creates a [`pastax.trajectory.Timeseries`][] from arrays of values and time points.
    """

    states: State
    _states_type: ClassVar = State
    times: Time #= eqx.field(static=True)  # TODO: not sure if this is correct
    length: int = eqx.field(static=True)

    _value: None = eqx.field(repr=False)
    _unit: None = eqx.field(repr=False)

    def __init__(self, states: State, times: Time, **_: Dict):
        """
        Initializes the [`pastax.trajectory.Timeseries`][] with given [`pastax.trajectory.State`][], [`pastax.trajectory.Time`][], and optional parameters.

        Parameters
        ----------
        states : Float[Array, "... time state"]
            The states of the [`pastax.trajectory.Timeseries`].
        times : Float[Array, "time"]
            The time points for the [`pastax.trajectory.Timeseries`].
        """
        super().__init__()
        self.states = states
        self.times = times
        self.length = times.value.shape[-1]

    @property
    def value(self) -> Float[Array, "... time state"]:
        """
        Returns the value of the [`pastax.trajectory.Timeseries`].

        Returns
        -------
        Float[Array, "... time state"]
            The value of the [`pastax.trajectory.Timeseries`].
        """
        return self.states.value

    @property
    def unit(self) -> Dict[Unit, int | float]:
        """
        Returns the unit of the [`pastax.trajectory.Timeseries`].

        Returns
        -------
        Dict[Unit, int | float]
            The unit of the [`pastax.trajectory.Timeseries`].
        """
        return self.states.unit

    @property
    def name(self) -> str:
        """
        Returns the name of the [`pastax.trajectory.Timeseries`].

        Returns
        -------
        name
            The name of the [`pastax.trajectory.Timeseries`].
        """
        return self.states.name
    
    def attach_name(self, name: str) -> Timeseries:
        """
        Attaches a name to the [`pastax.trajectory.Timeseries`].

        Parameters
        ----------
        name : str
            The name to attach to the [`pastax.trajectory.Timeseries`].

        Returns
        -------
        Timeseries
            A new [`pastax.trajectory.Timeseries`][] with the attached name.
        """
        return self.__class__(self.states.value, self.times.value, unit=self.unit, name=name)
    
    def euclidean_distance(self, other: Timeseries | ArrayLike) -> Timeseries:
        """
        Computes the Euclidean distance between this timeseries and another timeseries.

        Parameters
        ----------
        other : Timeseries | ArrayLike
            The other [`pastax.trajectory.Timeseries`][] to compute the distance to.

        Returns
        -------
        Timeseries
            The Euclidean distance between the two [`pastax.trajectory.Timeseries`].
        """
        if isinstance(other, Timeseries):
            other = other.states
        
        res = eqx.filter_vmap(lambda p1, p2: p1.euclidean_distance(p2))(self.states, other)

        return Timeseries.from_array(res.value, self.times.value, self.unit, name="Euclidean distance")

    def map(self, func: Callable[[State], Unitful | ArrayLike]) -> Timeseries:
        """
        Applies a function to each [`pastax.trajectory.State`][] in the [`pastax.trajectory.Timeseries`].

        Parameters
        ----------
        func : Callable[[State], Unitful | ArrayLike]
            The function to apply to each [`pastax.trajectory.State`][].

        Returns
        -------
        Timeseries
            The result of applying the function to each [`pastax.trajectory.State`][].
        """
        unit = {}
        res = eqx.filter_vmap(func, in_axes=_in_axes_func)(self.states)

        if isinstance(res, Unitful):
            unit = res.unit
            res = res.value

        return Timeseries.from_array(res, self.times.value, unit)

    def to_xarray(self) -> xr.Dataset:
        """
        Converts the [`pastax.trajectory.Timeseries`][] to a `xarray.Dataset`.

        Returns
        -------
        xr.Dataset
            The corresponding `xarray.Dataset`.
        """
        da = self._to_dataarray()
        ds = da.to_dataset()

        return ds
    
    @classmethod
    def from_array(
        cls, 
        values: Float[Array, "time state"], 
        times: Float[Array, "time"],
        unit: Unit | Dict[Unit, int | float] = {},
        name: str = None,
        **kwargs: Dict
    ) -> Timeseries:
        """
        Creates a [`pastax.trajectory.Timeseries`][] from arrays of values and time points.

        Parameters
        ----------
        values : Float[Array, "time state"]
            The array of values for the timeseries.
        times : Float[Array, "time"]
            The time points for the timeseries.
        unit : Unit | Dict[Unit, int | float], optional
            The unit of the timeseries, defaults to an empty Dict.
        name : str, optional
            The name of the timeseries, defaults to None.
        **kwargs : Dict
            Additional keyword arguments.

        Returns
        -------
        Timeseries
            The corresponding [`pastax.trajectory.Timeseries`][].
        """
        values = jnp.asarray(values, dtype=float)
        times = jnp.asarray(times, dtype=float)
        
        values = eqx.filter_vmap(
            lambda value: cls._states_type(value, unit=unit, name=name),
            out_axes=eqx._vmap_pmap.if_mapped(0)
        )(values)

        times = eqx.filter_vmap(
            lambda time: Time(time),
            out_axes=eqx._vmap_pmap.if_mapped(0)
        )(times)

        return cls(values, times, **kwargs)

    def _to_dataarray(self) -> xr.DataArray:
        da = xr.DataArray(
            data=self.states.value,
            dims=["time"],
            coords={"time": self.times.to_datetime()},
            name=self.name,
            attrs={"units": units_to_str(self.unit)}
        )

        return da
