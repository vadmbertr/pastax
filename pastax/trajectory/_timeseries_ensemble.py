from __future__ import annotations
from typing import Dict, Callable, ClassVar

import equinox as eqx
from jaxtyping import Array, ArrayLike, Float
import numpy as np
import xarray as xr

from ..utils.unit import Unit, units_to_str
from ._state import State
from .state import Time
from ._timeseries import Timeseries
from ._unitful import Unitful


def _in_axes_func(leaf):
    axes = None
    if eqx.is_array(leaf) and leaf.ndim > 1:
        axes = 0
    return axes


class TimeseriesEnsemble(Unitful):
    """
    Class representing `pastax.TimeseriesEnsemble`.

    Attributes
    ----------
    members : Timeseries
        The members of the `pastax.TimeseriesEnsemble`.
    _members_type : ClassVar
        The type of the members in the `pastax.TimeseriesEnsemble`.
    size : int
        The number of members in the `pastax.TimeseriesEnsemble`.

    Methods
    -------
    __init__(members)
        Initializes the `pastax.TimeseriesEnsemble` with [`pastax.Timeseries`][] members.
    value
        Returns the value of the `pastax.TimeseriesEnsemble`.
    states
        Returns the [`pastax.State`][] of the `pastax.TimeseriesEnsemble`.
    times
        Returns the [`pastax.Time`][] points of the `pastax.TimeseriesEnsemble`.
    unit
        Returns the unit of the `pastax.TimeseriesEnsemble`.
    name
        Returns the name of the `pastax.TimeseriesEnsemble`.
    length
        Returns the length of the `pastax.TimeseriesEnsemble`.
    attach_name(name)
        Attaches a name to the `pastax.TimeseriesEnsemble`.
    crps(other, distance_func=Timeseries.euclidean_distance)
        Computes the Continuous Ranked Probability Score (CRPS) for the `pastax.TimeseriesEnsemble`.
    ensemble_dispersion(distance_func=Timeseries.euclidean_distance)
        Computes the `pastax.TimeseriesEnsemble` dispersion.
    map(func)
        Applies a function to each [`pastax.Timeseries`][] of the `pastax.TimeseriesEnsemble`.
    from_array(values, times, unit={}, name=None, **kwargs)
        Creates a `pastax.TimeseriesEnsemble` from an array of values and time points.
    to_dataarray()
        Converts the `pastax.TimeseriesEnsemble` [`pastax.State`][] to a `xarray.DataArray`.
    to_dataset()
        Converts the `pastax.TimeseriesEnsemble` to a `xarray.Dataset`.
    """

    members: Timeseries
    _members_type: ClassVar = Timeseries
    size: int = eqx.field(static=True)

    _value: None = eqx.field(repr=False)
    _unit: None = eqx.field(repr=False)

    def __init__(self, members: Timeseries):
        """
        Initializes the `pastax.TimeseriesEnsemble` with [`pastax.Timeseries`][] members.

        Parameters
        ----------
        members : Timeseries
            The members of the `pastax.TimeseriesEnsemble`.
        """
        super().__init__()
        self.members = members
        self.size = members.states.value.shape[0]

    @property
    def value(self) -> Float[Array, "member time state"]:
        """
        Returns the value of the `pastax.TimeseriesEnsemble`.

        Returns
        -------
        Float[Array, "... member time state"]
            The value of the `pastax.TimeseriesEnsemble`.
        """
        return self.members.value

    @property
    def states(self) -> State:
        """
        Returns the states of the `pastax.TimeseriesEnsemble`.

        Returns
        -------
        State
            The states of the `pastax.TimeseriesEnsemble`.
        """
        return self.members.states

    @property
    def times(self) -> Time:
        """
        Returns the time points of the `pastax.TimeseriesEnsemble`.

        Returns
        -------
        Time
            The time points of the `pastax.TimeseriesEnsemble`.
        """
        return self.members.times

    @property
    def unit(self) -> Dict[Unit, int | float]:
        """
        Returns the unit of the `pastax.TimeseriesEnsemble`.

        Returns
        -------
        Dict[Unit, int | float]
            The unit of the `pastax.TimeseriesEnsemble`.
        """
        return self.members.unit

    @property
    def name(self) -> str:
        """
        Returns the name of the `pastax.TimeseriesEnsemble`.

        Returns
        -------
        str
            The name of the `pastax.TimeseriesEnsemble`.
        """
        return self.members.name

    @property
    def length(self) -> int:
        """
        Returns the length of the `pastax.TimeseriesEnsemble`.

        Returns
        -------
        int
            The length of the `pastax.TimeseriesEnsemble`.
        """
        return self.members.length

    def attach_name(self, name: str) -> TimeseriesEnsemble:
        """
        Attaches a name to the `pastax.TimeseriesEnsemble`.

        Parameters
        ----------
        name : str
            The name to attach to the `pastax.TimeseriesEnsemble`.

        Returns
        -------
        TimeseriesEnsemble
            A new `pastax.TimeseriesEnsemble` with the attached name.
        """
        return self.__class__(self.states.value, self.times.value, unit=self.unit, name=name)
    
    def crps(
        self,
        other: Timeseries,
        distance_func: Callable[[Timeseries, Timeseries], Unitful | ArrayLike] = Timeseries.euclidean_distance
    ) -> Timeseries:
        """
        Computes the Continuous Ranked Probability Score (CRPS) for the `pastax.TimeseriesEnsemble`.

        Parameters
        ----------
        other : Timeseries
            The other timeseries to compare against.
        distance_func : Callable[[Timeseries, Timeseries], Unitful | ArrayLike], optional
            The distance function to use (default is Timeseries.euclidean_distance).

        Returns
        -------
        Timeseries
            The CRPS for the `pastax.TimeseriesEnsemble`.
        """
        biases = self.map(lambda member: distance_func(other, member))
        bias = biases.mean(axis=0)

        n_members = self.size
        dispersion = self.ensemble_dispersion(distance_func)
        dispersion /= n_members * (n_members - 1)

        crps = bias - dispersion

        return Timeseries.from_array(crps.value, self.times.value, crps.unit, name=f"{biases.name} (CRPS)")

    def ensemble_dispersion(
        self,
        distance_func: Callable[[Timeseries, Timeseries], Unitful | ArrayLike] = Timeseries.euclidean_distance
    ) -> Timeseries:
        """
        Computes the `pastax.TimeseriesEnsemble` dispersion.

        Parameters
        ----------
        distance_func : Callable[[Timeseries, Timeseries], Unitful | ArrayLike], optional
            The distance function to use (default is Timeseries.euclidean_distance).

        Returns
        -------
        Timeseries
            The `pastax.TimeseriesEnsemble` dispersion.
        """
        distances = self.map(lambda member1: self.map(lambda member2: distance_func(member1, member2)))
        # divide by 2 as pairs are duplicated (JAX efficient computation way)
        dispersion = distances.sum((-3, -2)) / 2

        return Timeseries.from_array(
            dispersion.value, self.times.value, dispersion.unit, name=f"{distances.name} (ensemble dispersion)"
        )

    def map(self, func: Callable[[Timeseries], Unitful | ArrayLike]) -> TimeseriesEnsemble:
        """
        Applies a function to each [`pastax.Timeseries`][] of the `pastax.TimeseriesEnsemble`.

        Parameters
        ----------
        func : Callable[[Timeseries], Unitful | ArrayLike]
            The function to apply to each [`pastax.Timeseries`][].

        Returns
        -------
        TimeseriesEnsemble
            The result of applying the function to each [`pastax.Timeseries`][].
        """
        unit = {}
        res = eqx.filter_vmap(func, in_axes=_in_axes_func)(self.members)

        if isinstance(res, Unitful):
            unit = res.unit
            res = res.value

        return TimeseriesEnsemble.from_array(res, self.times.value, unit)

    @classmethod
    def from_array(
        cls,
        values: Float[Array, "member time state"],
        times: Float[Array, "time"],
        unit: Unit | Dict[Unit, int | float] = {},
        name: str = None,
        **kwargs: Dict
    ) -> TimeseriesEnsemble:
        """
        Creates a `pastax.TimeseriesEnsemble` from an array of values and time points.

        Parameters
        ----------
        values : Float[Array, "member time state"]
            The values for the members of the ensemble.
        times : Float[Array, "time"]
            The time points for the timeseries.
        name : str, optional
            The name of the timeseries (default is None).
        unit : Unit | Dict[Unit, int | float], optional
            The unit of the timeseries (default is {}).
        **kwargs : Dict
            Additional keyword arguments.

        Returns
        -------
        TimeseriesEnsemble
            The `pastax.TimeseriesEnsemble` created from the array of values and time points.
        """
        members = eqx.filter_vmap(
            lambda s: cls._members_type.from_array(s, times, unit=unit, name=name, **kwargs),
            out_axes=eqx._vmap_pmap.if_mapped(0)
        )(values)

        return cls(members)

    def to_dataarray(self) -> xr.DataArray:
        """
        Converts the `pastax.TimeseriesEnsemble` [`pastax.State`][] to a `xarray.DataArray`.

        Returns
        -------
        xr.DataArray
            An `xarray.DataArray` containing the `pastax.TimeseriesEnsemble` [`pastax.State`][].
        """
        da = xr.DataArray(
            data=self.states.value,
            dims=["member", "time"],
            coords={
                "member": np.arange(self.size),
                "time": self.members.times.to_datetime()
            },
            name=self.name,
            attrs={"units": units_to_str(self.unit)}
        )

        return da

    def to_dataset(self) -> xr.Dataset:
        """
        Converts the `pastax.TimeseriesEnsemble` [`pastax.State`][] to a `xarray.DataArray`.

        Returns
        -------
        xr.Dataset
            A `xarray.DataArray` containing the `pastax.TimeseriesEnsemble` [`pastax.State`][].
        """
        da = self.to_dataarray()
        ds = da.to_dataset()

        return ds
