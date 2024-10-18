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
    Base class representing an ensemble of timeseries data.

    Attributes
    ----------
    members : Timeseries
        The members of the timeseries ensemble.
    _members_type : ClassVar
        The type of the members in the ensemble.
    size : int
        The number of members in the ensemble.

    Methods
    -------
    __init__(values, times, unit={}, name=None, **kwargs)
        Initializes the TimeseriesEnsemble with given values, times, and optional parameters.
    value
        Returns the value of the ensemble.
    states
        Returns the states of the ensemble.
    times
        Returns the time points of the ensemble.
    unit
        Returns the unit of the ensemble.
    name
        Returns the name of the ensemble.
    length
        Returns the length of the ensemble.
    attach_name(name)
        Attaches a name to the ensemble.
    crps(other, distance_func=Timeseries.euclidean_distance)
        Computes the Continuous Ranked Probability Score (CRPS) for the ensemble.
    ensemble_dispersion(distance_func=Timeseries.euclidean_distance)
        Computes the ensemble dispersion.
    map(func)
        Applies a function to each timeseries of the ensemble.
    from_array(values, times, unit={}, name=None, **kwargs)
        Creates a TimeseriesEnsemble from an array of values and time points.
    to_dataarray()
        Converts the ensemble states to a xarray DataArray.
    to_dataset()
        Converts the ensemble to a xarray Dataset.
    __sub__(other)
        Subtracts another ensemble, timeseries, state or array like from this ensemble.
    """

    members: Timeseries
    _members_type: ClassVar = Timeseries
    size: int = eqx.field(static=True)
    value: None = eqx.field(static=True, default_factory=lambda: None)
    unit: None = eqx.field(static=True, default_factory=lambda: None)

    def __init__(
        self,
        members: Timeseries
    ):
        """
        Initializes the TimeseriesEnsemble with members.

        Parameters
        ----------
        members : Timeseries
            The members of the ensemble.
        """
        self.members = members
        self.size = members.states.value.shape[0]

    @property
    def value(self) -> Float[Array, "member time state"]:
        """
        Returns the value of the ensemble.

        Returns
        -------
        Float[Array, "... member time state"]
            The value of the ensemble.
        """
        return self.members.value

    @property
    def states(self) -> State:
        """
        Returns the states of the ensemble.

        Returns
        -------
        State
            The states of the ensemble.
        """
        return self.members.states

    @property
    def times(self) -> Time:
        """
        Returns the time points of the ensemble.

        Returns
        -------
        Time
            The time points of the ensemble.
        """
        return self.members.times

    @property
    def unit(self) -> Dict[Unit, int | float]:
        """
        Returns the unit of the ensemble.

        Returns
        -------
        Dict[Unit, int | float]
            The unit of the ensemble.
        """
        return self.members.unit

    @property
    def name(self) -> str:
        """
        Returns the name of the ensemble.

        Returns
        -------
        str
            The name of the ensemble.
        """
        return self.members.name

    @property
    def length(self) -> int:
        """
        Returns the length of the ensemble.

        Returns
        -------
        int
            The length of the ensemble.
        """
        return self.members.length

    def attach_name(self, name: str) -> TimeseriesEnsemble:
        """
        Attaches a name to the ensemble.

        Parameters
        ----------
        name : str
            The name to attach to the ensemble.

        Returns
        -------
        TimeseriesEnsemble
            A new ensemble with the attached name.
        """
        return self.__class__(self.states.value, self.times.value, unit=self.unit, name=name)
    
    def crps(
        self,
        other: Timeseries,
        distance_func: Callable[[Timeseries, Timeseries], Unitful | ArrayLike] = Timeseries.euclidean_distance
    ) -> Timeseries:
        """
        Computes the Continuous Ranked Probability Score (CRPS) for the ensemble.

        Parameters
        ----------
        other : Timeseries
            The other timeseries to compare against.
        distance_func : Callable[[Timeseries, Timeseries], Unitful | ArrayLike], optional
            The distance function to use (default is Timeseries.euclidean_distance).

        Returns
        -------
        Timeseries
            The CRPS for the ensemble.
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
        Computes the ensemble dispersion.

        Parameters
        ----------
        distance_func : Callable[[Timeseries, Timeseries], Unitful | ArrayLike], optional
            The distance function to use (default is Timeseries.euclidean_distance).

        Returns
        -------
        Timeseries
            The ensemble dispersion.
        """
        distances = self.map(lambda member1: self.map(lambda member2: distance_func(member1, member2)))
        # divide by 2 as pairs are duplicated (JAX efficient computation way)
        dispersion = distances.sum((-3, -2)) / 2

        return Timeseries.from_array(
            dispersion.value, self.times.value, dispersion.unit, name=f"{distances.name} (ensemble dispersion)"
        )

    def map(self, func: Callable[[Timeseries], Unitful | ArrayLike]) -> TimeseriesEnsemble:
        """
        Applies a function to each member of the ensemble.

        Parameters
        ----------
        func : Callable[[Timeseries], Unitful | ArrayLike]
            The function to apply to each member.

        Returns
        -------
        TimeseriesEnsemble
            The result of applying the function to each timeseries.
        """
        unit = {}
        res = eqx.filter_vmap(
            func, 
            in_axes=_in_axes_func,
            out_axes=eqx._vmap_pmap.if_mapped(0)  # TODO: not sure if this is correct (remove?)
        )(self.members)

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
        Creates a TimeseriesEnsemble from an array of values and time points.

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
            The TimeseriesEnsemble created from the array of values and time points.
        """
        members = eqx.filter_vmap(
            lambda s: cls._members_type.from_array(s, times, unit=unit, name=name, **kwargs),
            out_axes=eqx._vmap_pmap.if_mapped(0)
        )(values)

        return cls(members)

    def to_dataarray(self) -> xr.DataArray:
        """
        Converts the ensemble states to an xarray DataArray.

        Returns
        -------
        xr.DataArray
            An xarray DataArray containing the ensemble states.
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
        Converts the ensemble states to an xarray Dataset.

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the ensemble states.
        """
        da = self.to_dataarray()
        ds = da.to_dataset()

        return ds
