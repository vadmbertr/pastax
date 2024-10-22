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
    Class representing a timeseries of states.

    Attributes
    ----------
    states : State
        The states of the timeseries.
    _states_type : ClassVar
        The type of the states in the timeseries.
    times : Time
        The time points of the timeseries.
    length : int
        The length of the timeseries.

    Methods
    -------
    __init__(states, times, **__)
        Initializes the Timeseries with given states, times, and optional parameters.
    value
        Returns the value of the timeseries.
    unit
        Returns the unit of the timeseries.
    name
        Returns the name of the timeseries.
    attach_name(name)
        Attaches a name to the timeseries.
    euclidean_distance(other)
        Computes the Euclidean distance between this timeseries and another timeseries.
    map(func)
        Applies a function to each state in the timeseries.
    from_array(values, times, unit={}, name=None, **kwargs)
        Creates a Timeseries from an array of values and time points.
    to_dataarray()
        Converts the timeseries states to an xarray DataArray.
    to_dataset()
        Converts the timeseries to a xarray Dataset.
    __sub__(other)
        Subtracts another ensemble, timeseries, state or array like from this timeseries.
    """

    states: State
    _states_type: ClassVar = State
    times: Time #= eqx.field(static=True)  # TODO: not sure if this is correct
    length: int = eqx.field(static=True)

    _value: None = eqx.field(repr=False)
    _unit: None = eqx.field(repr=False)

    def __init__(
        self,
        states: State,
        times: Time,
        **_: Dict
    ):
        """
        Initializes the Timeseries with given states, times, and optional parameters.

        Parameters
        ----------
        states : Float[Array, "... time state"]
            The states of the timeseries.
        times : Float[Array, "time"]
            The time points for the timeseries.
        """
        super().__init__()
        self.states = states
        self.times = times
        self.length = times.value.shape[-1]

    @property
    def value(self) -> Float[Array, "... time state"]:
        """
        Returns the value of the timeseries.

        Returns
        -------
        Float[Array, "... time state"]
            The value of the timeseries.
        """
        return self.states.value

    @property
    def unit(self) -> Dict[Unit, int | float]:
        """
        Returns the unit of the timeseries.

        Returns
        -------
        Dict[Unit, int | float]
            The unit of the timeseries.
        """
        return self.states.unit

    @property
    def name(self) -> str:
        """
        Returns the name of the timeseries.

        Returns
        -------
        name
            The name of the timeseries.
        """
        return self.states.name
    
    def attach_name(self, name: str) -> Timeseries:
        """
        Attaches a name to the timeseries.

        Parameters
        ----------
        name : str
            The name to attach to the timeseries.

        Returns
        -------
        Timeseries
            A new timeseries with the attached name.
        """
        return self.__class__(self.states.value, self.times.value, unit=self.unit, name=name)
    
    def euclidean_distance(self, other: Timeseries | ArrayLike) -> Timeseries:
        """
        Computes the Euclidean distance between this timeseries and another timeseries.

        Parameters
        ----------
        other : Timeseries | ArrayLike
            The other timeseries to compute the distance to.

        Returns
        -------
        Timeseries
            The Euclidean distance between the two timeseries.
        """
        if isinstance(other, Timeseries):
            other = other.states
        
        res = eqx.filter_vmap(lambda p1, p2: p1.euclidean_distance(p2))(self.states, other)

        return Timeseries.from_array(res.value, self.times.value, self.unit, name="Euclidean distance")

    def map(self, func: Callable[[State], Unitful | ArrayLike]) -> Timeseries:
        """
        Applies a function to each state in the timeseries.

        Parameters
        ----------
        func : Callable[[State], Unitful | ArrayLike]
            The function to apply to each state.

        Returns
        -------
        Timeseries
            The result of applying the function to each state.
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
        Creates a timeseries from an array of values and time points.

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
            The timeseries created from the array of values and time points.
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
        Converts the timeseries states to an xarray DataArray.

        Returns
        -------
        xr.DataArray
            An xarray DataArray containing the timeseries states.
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
        Converts the timeseries states to an xarray Dataset.

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the timeseries states.
        """
        da = self.to_dataarray()
        ds = da.to_dataset()

        return ds
