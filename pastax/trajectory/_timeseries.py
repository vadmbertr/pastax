from __future__ import annotations
from typing import Callable, ClassVar, Dict

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float
import xarray as xr

from ..utils.unit import Unit, units_to_str
from ._state import State
from .state import Time
from ._unitful import Unitful


def _in_axes_func(leaf):
    axes = None
    if eqx.is_array(leaf) and leaf.ndim > 0:
        axes = 0
    return axes


class Timeseries(Unitful):
    """
    Class representing a `pastax.Timeseries`.

    Attributes
    ----------
    states : State
        The [`pastax.State`][] of the `pastax.Timeseries`.
    _states_type : ClassVar
        The type of the [`pastax.State`][] in the `pastax.Timeseries`.
    times : Time
        The [`pastax.Time`][] points of the `pastax.Timeseries`.
    length : int
        The length of the `pastax.Timeseries`.

    Methods
    -------
    __init__(states, times, **__)
        Initializes the `pastax.Timeseries` with given [`pastax.State`][], [`pastax.Time`][], and optional parameters.
    value
        Returns the value of the `pastax.Timeseries`.
    unit
        Returns the unit of the `pastax.Timeseries`.
    name
        Returns the name of the `pastax.Timeseries`.
    attach_name(name)
        Attaches a name to the `pastax.Timeseries`.
    euclidean_distance(other)
        Computes the Euclidean distance between this `pastax.Timeseries` and another `pastax.Timeseries`.
    map(func)
        Applies a function to each [`pastax.State`][] in the `pastax.Timeseries`.
    from_array(values, times, unit={}, name=None, **kwargs)
        Creates a `pastax.Timeseries` from an array of values and time points.
    to_dataarray()
        Converts the `pastax.Timeseries` states to an xarray DataArray.
    to_dataset()
        Converts the `pastax.Timeseries` to a xarray Dataset.
    """

    states: State
    _states_type: ClassVar = State
    times: Time #= eqx.field(static=True)  # TODO: not sure if this is correct
    length: int = eqx.field(static=True)

    _value: None = eqx.field(repr=False)
    _unit: None = eqx.field(repr=False)

    def __init__(self, states: State, times: Time, **_: Dict):
        """
        Initializes the `pastax.Timeseries` with given [`pastax.State`][], [`pastax.Time`][], and optional parameters.

        Parameters
        ----------
        states : Float[Array, "... time state"]
            The states of the `pastax.Timeseries`.
        times : Float[Array, "time"]
            The time points for the `pastax.Timeseries`.
        """
        super().__init__()
        self.states = states
        self.times = times
        self.length = times.value.shape[-1]

    @property
    def value(self) -> Float[Array, "... time state"]:
        """
        Returns the value of the `pastax.Timeseries`.

        Returns
        -------
        Float[Array, "... time state"]
            The value of the `pastax.Timeseries`.
        """
        return self.states.value

    @property
    def unit(self) -> Dict[Unit, int | float]:
        """
        Returns the unit of the `pastax.Timeseries`.

        Returns
        -------
        Dict[Unit, int | float]
            The unit of the `pastax.Timeseries`.
        """
        return self.states.unit

    @property
    def name(self) -> str:
        """
        Returns the name of the `pastax.Timeseries`.

        Returns
        -------
        name
            The name of the `pastax.Timeseries`.
        """
        return self.states.name
    
    def attach_name(self, name: str) -> Timeseries:
        """
        Attaches a name to the `pastax.Timeseries`.

        Parameters
        ----------
        name : str
            The name to attach to the `pastax.Timeseries`.

        Returns
        -------
        Timeseries
            A new `pastax.Timeseries` with the attached name.
        """
        return self.__class__(self.states.value, self.times.value, unit=self.unit, name=name)
    
    def euclidean_distance(self, other: Timeseries | ArrayLike) -> Timeseries:
        """
        Computes the Euclidean distance between this timeseries and another timeseries.

        Parameters
        ----------
        other : Timeseries | ArrayLike
            The other `pastax.Timeseries` to compute the distance to.

        Returns
        -------
        Timeseries
            The Euclidean distance between the two `pastax.Timeseries`.
        """
        if isinstance(other, Timeseries):
            other = other.states
        
        res = eqx.filter_vmap(lambda p1, p2: p1.euclidean_distance(p2))(self.states, other)

        return Timeseries.from_array(res.value, self.times.value, self.unit, name="Euclidean distance")

    def map(self, func: Callable[[State], Unitful | ArrayLike]) -> Timeseries:
        """
        Applies a function to each [`pastax.State`][] in the `pastax.Timeseries`.

        Parameters
        ----------
        func : Callable[[State], Unitful | ArrayLike]
            The function to apply to each [`pastax.State`][].

        Returns
        -------
        Timeseries
            The result of applying the function to each [`pastax.State`][].
        """
        unit = {}
        res = eqx.filter_vmap(func, in_axes=_in_axes_func)(self.states)

        if isinstance(res, Unitful):
            unit = res.unit
            res = res.value

        return Timeseries.from_array(res, self.times.value, unit)
    
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
        Creates a `pastax.Timeseries` from an array of values and time points.

        Parameters
        ----------
        values : Float[Array, "time state"]
            The array of values for the timeseries.
        times : Float[Array, "time"]
            The time points for the timeseries.
        unit : Unit | Dict[Unit, int | float], optional
            The unit of the timeseries (default is an empty Dict).
        name : str, optional
            The name of the timeseries (default is None).
        **kwargs : Dict
            Additional keyword arguments.

        Returns
        -------
        Timeseries
            The `pastax.Timeseries` created from the array of values and time points.
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

    def to_dataarray(self) -> xr.DataArray:
        """
        Converts the `pastax.Timeseries` [`pastax.State`][] to a `xarray.DataArray`.

        Returns
        -------
        xr.DataArray
            A `xarray.DataArray` containing the `pastax.Timeseries` [`pastax.State`][].
        """
        da = xr.DataArray(
            data=self.states.value,
            dims=["time"],
            coords={"time": self.times.to_datetime()},
            name=self.name,
            attrs={"units": units_to_str(self.unit)}
        )

        return da

    def to_dataset(self) -> xr.Dataset:
        """
        Converts the `pastax.Timeseries` [`pastax.State`][] to a `xarray.Dataset`.

        Returns
        -------
        xr.Dataset
            A `xarray.Dataset` containing the `pastax.Timeseries` [`pastax.State`][].
        """
        da = self.to_dataarray()
        ds = da.to_dataset()

        return ds
