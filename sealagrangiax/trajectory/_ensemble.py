from __future__ import annotations
from numbers import Number
from typing import Callable, ClassVar

import equinox as eqx
from jaxtyping import Array, Float, Int
import numpy as np
import xarray as xr

from ..utils import UNIT
from ._state import WHAT
from ._timeseries import Timeseries


class TimeseriesEnsemble(eqx.Module):
    """
    Base class representing an ensemble of timeseries data.

    Attributes
    ----------
    _members : Timeseries
        The members of the timeseries ensemble.
    _members_type : ClassVar
        The type of the members in the ensemble.
    size : int
        The number of members in the ensemble.

    Methods
    -------
    __init__(values, times, what=WHAT.unknown, unit=UNIT.dimensionless, **kwargs)
        Initializes the TimeseriesEnsemble with given values, times, and optional parameters.
    length
        Returns the length of the timeseries.
    states
        Returns the states of the timeseries.
    times
        Returns the time points of the timeseries.
    unit
        Returns the unit of the timeseries.
    what
        Returns the type of the timeseries.
    crps(other, distance_func=Timeseries.euclidean_distance)
        Computes the Continuous Ranked Probability Score (CRPS) for the ensemble.
    ensemble_dispersion(distance_func=Timeseries.euclidean_distance)
        Computes the ensemble dispersion.
    map(func)
        Applies a function to each timeseries of the ensemble.
    mean
        Computes the mean of the ensemble along the `member` dimension.
    __sub__(other)
        Subtracts another timeseries or value from the ensemble.
    """

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
        """
        Initializes the TimeseriesEnsemble with given values, times, and optional parameters.

        Parameters
        ----------
        values : Float[Array, "member time state"]
            The values for the members of the ensemble.
        times : Int[Array, "time"]
            The time points for the timeseries.
        what : WHAT, optional
            The type of the timeseries (default is WHAT.unknown).
        unit : UNIT, optional
            The unit of the timeseries (default is UNIT.dimensionless).
        **kwargs
            Additional keyword arguments.
        """
        self._members = eqx.filter_vmap(
            lambda s: self._members_type(s, times, what=what, unit=unit, **kwargs),
            out_axes=eqx._vmap_pmap.if_mapped(0)
        )(values)
        self.size = values.shape[0]

    @property
    def length(self) -> int:
        """
        Returns the length of the timeseries.

        Returns
        -------
        int
            The length of the timeseries.
        """
        return self._members.length

    @property
    def states(self) -> Float[Array, "member time state"]:
        """
        Returns the states of the timeseries.

        Returns
        -------
        Float[Array, "member time state"]
            The states of the timeseries.
        """
        return self._members.states

    @property
    def times(self) -> Int[Array, "time"]:
        """
        Returns the time points of the timeseries.

        Returns
        -------
        Int[Array, "time"]
            The time points of the timeseries.
        """
        return self._members.times

    @property
    def unit(self) -> UNIT:
        """
        Returns the unit of the timeseries.

        Returns
        -------
        UNIT
            The unit of the timeseries.
        """
        return self._members.unit

    @property
    def what(self) -> WHAT:
        """
        Returns the type of the timeseries.

        Returns
        -------
        WHAT
            The type of the timeseries.
        """
        return self._members.what

    @eqx.filter_jit
    def crps(
        self,
        other: Timeseries,
        distance_func: Callable[[Timeseries, Timeseries], Float[Array, "time"]] = Timeseries.euclidean_distance
    ) -> Float[Array, "time"] | Timeseries:
        """
        Computes the Continuous Ranked Probability Score (CRPS) for the ensemble.

        Parameters
        ----------
        other : Timeseries
            The other timeseries to compare against.
        distance_func : Callable[[Timeseries, Timeseries], Float[Array, "time"]], optional
            The distance function to use (default is Timeseries.euclidean_distance).

        Returns
        -------
        Float[Array, "time"] | Timeseries
            The CRPS for the ensemble.
        """
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
        """
        Computes the ensemble dispersion.

        Parameters
        ----------
        distance_func : Callable[[Timeseries, Timeseries], Float[Array, "times"]], optional
            The distance function to use (default is Timeseries.euclidean_distance).

        Returns
        -------
        Float[Array, "times"] | Timeseries
            The ensemble dispersion.
        """
        distances = self.map(lambda member1: self.map(lambda member2: distance_func(member1, member2)))
        # divide by 2 as pairs are duplicated (JAX efficient computation way)
        dispersion = distances.sum((0, 1)) / 2

        return dispersion

    @eqx.filter_jit
    def map(self, func: Callable[[Timeseries], Float[Array, "..."]]) -> Float[Array, "member ..."]:
        """
        Applies a function to each timeseries of the ensemble.

        Parameters
        ----------
        func : Callable[[Timeseries], Float[Array, "..."]]
            The function to apply to each timeseries.

        Returns
        -------
        Float[Array, "member ..."]
            The result of applying the function to each timeseries.
        """
        def axes_func(leaf):
            axes = None
            if eqx.is_array(leaf) and leaf.ndim > 1:
                axes = 0
            return axes

        return eqx.filter_vmap(func, in_axes=axes_func)(self._members)

    @eqx.filter_jit
    def mean(self) -> Float[Array, "time state"]:
        """
        Computes the mean of the ensemble along the `member` dimension.

        Returns
        -------
        Float[Array, "time state"]
            The timeseries mean of the ensemble.
        """
        return self.states.mean(axis=0)

    def to_dataarray(self) -> xr.DataArray:
        """
        Converts the ensemble states to an xarray DataArray.

        Returns
        -------
        xr.DataArray
            An xarray DataArray containing the ensemble states.
        """
        da = xr.DataArray(
            data=self.states,
            dims=["member", "time"],
            coords={
                "member": np.arange(self.size),
                "time": self._members._times.to_datetime()  # noqa
            },
            name=WHAT[self.what],
            attrs={"units": UNIT[self.unit]}
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

    @eqx.filter_jit
    def __sub__(self, other: Timeseries | Float[Array, "time state"] | Number) -> Float[Array, "member time state"]:
        """
        Subtracts another timeseries or value from the ensemble.

        Parameters
        ----------
        other : Timeseries | Float[Array, "time state"] | Number
            The other timeseries or value to subtract.

        Returns
        -------
        Float[Array, "member time state"]
            The result of the subtraction.
        """
        return self.map(lambda timeseries: timeseries.__sub__(other))
