from __future__ import annotations

import dataclasses
from typing import Any, Callable, ClassVar, Optional

import equinox as eqx
import jax.interpreters.batching as batching
import jax.interpreters.pxla as pxla
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Array, Float

from ..utils._unit import Unit, units_to_str
from ._state import State
from ._states import Time
from ._timeseries import Timeseries
from ._unitful import Unitful


# copy-paste from equinox
@dataclasses.dataclass(frozen=True)
class _if_mapped:
    axis: int

    def __call__(self, x: Any) -> Optional[int]:
        if isinstance(x, batching.BatchTracer):
            if x.batch_dim is batching.not_mapped:
                return None
            else:
                return self.axis
        elif isinstance(x, pxla.MapTracer):
            return self.axis
        else:
            return None


class TimeseriesEnsemble(Unitful):
    """
    Class representing [`pastax.trajectory.TimeseriesEnsemble`].

    Attributes
    ----------
    members : Timeseries
        The members of the [`pastax.trajectory.TimeseriesEnsemble`].
    size : int
        The number of members in the [`pastax.trajectory.TimeseriesEnsemble`].

    Methods
    -------
    __init__(members)
        Initializes the [`pastax.trajectory.TimeseriesEnsemble`][] with [`pastax.trajectory.Timeseries`][] members.
    value
        Returns the value of the [`pastax.trajectory.TimeseriesEnsemble`].
    states
        Returns the [`pastax.trajectory.State`][] of the [`pastax.trajectory.TimeseriesEnsemble`].
    times
        Returns the [`pastax.trajectory.Time`][] points of the [`pastax.trajectory.TimeseriesEnsemble`].
    unit
        Returns the unit of the [`pastax.trajectory.TimeseriesEnsemble`].
    name
        Returns the name of the [`pastax.trajectory.TimeseriesEnsemble`].
    length
        Returns the length of the [`pastax.trajectory.TimeseriesEnsemble`].
    attach_name(name)
        Attaches a name to the [`pastax.trajectory.TimeseriesEnsemble`].
    crps(other, metric_func)
        Computes the Continuous Ranked Probability Score (CRPS) for the [`pastax.trajectory.TimeseriesEnsemble`].
    ensemble_dispersion(metric_func)
        Computes the [`pastax.trajectory.TimeseriesEnsemble`][] dispersion.
    map(func)
        Applies a function to each [`pastax.trajectory.Timeseries`][] of the [`pastax.trajectory.TimeseriesEnsemble`].
    to_xarray()
        Converts the [`pastax.trajectory.TimeseriesEnsemble`][] to a `xarray.Dataset`.
    from_array(values, times, unit={}, name=None, **kwargs)
        Creates a [`pastax.trajectory.TimeseriesEnsemble`][] from arrays of values and time points.
    """

    members: Timeseries
    _members_type: ClassVar = Timeseries
    size: int = eqx.field(static=True)

    _value: None = eqx.field(repr=False)
    _unit: None = eqx.field(repr=False)

    def __init__(self, members: Timeseries):
        """
        Initializes the [`pastax.trajectory.TimeseriesEnsemble`][] with [`pastax.trajectory.Timeseries`][] members.

        Parameters
        ----------
        members : Timeseries
            The members of the [`pastax.trajectory.TimeseriesEnsemble`].
        """
        super().__init__()
        self.members = members
        self.size = members.states.value.shape[0]

    @property
    def value(self) -> Float[Array, "member time state"]:
        """
        Returns the value of the [`pastax.trajectory.TimeseriesEnsemble`].

        Returns
        -------
        Float[Array, "... member time state"]
            The value of the [`pastax.trajectory.TimeseriesEnsemble`].
        """
        return self.members.value

    @property
    def states(self) -> State:
        """
        Returns the states of the [`pastax.trajectory.TimeseriesEnsemble`].

        Returns
        -------
        State
            The states of the [`pastax.trajectory.TimeseriesEnsemble`].
        """
        return self.members.states

    @property
    def times(self) -> Time:
        """
        Returns the time points of the [`pastax.trajectory.TimeseriesEnsemble`].

        Returns
        -------
        Time
            The time points of the [`pastax.trajectory.TimeseriesEnsemble`].
        """
        return self.members.times

    @property
    def unit(self) -> dict[Unit, int | float]:
        """
        Returns the unit of the [`pastax.trajectory.TimeseriesEnsemble`].

        Returns
        -------
        dict[Unit, int | float]
            The unit of the [`pastax.trajectory.TimeseriesEnsemble`].
        """
        return self.members.unit

    @property
    def name(self) -> str | None:
        """
        Returns the name of the [`pastax.trajectory.TimeseriesEnsemble`].

        Returns
        -------
        str | None
            The name of the [`pastax.trajectory.TimeseriesEnsemble`].
        """
        return self.members.name

    @property
    def length(self) -> int:
        """
        Returns the length of the [`pastax.trajectory.TimeseriesEnsemble`].

        Returns
        -------
        int
            The length of the [`pastax.trajectory.TimeseriesEnsemble`].
        """
        return self.members.length

    def attach_name(self, name: str) -> TimeseriesEnsemble:
        """
        Attaches a name to the [`pastax.trajectory.TimeseriesEnsemble`].

        Parameters
        ----------
        name : str
            The name to attach to the [`pastax.trajectory.TimeseriesEnsemble`].

        Returns
        -------
        TimeseriesEnsemble
            A new [`pastax.trajectory.TimeseriesEnsemble`][] with the attached name.
        """
        return TimeseriesEnsemble.from_array(self.states.value, self.times.value, unit=self.unit, name=name)

    def crps(
        self,
        other: Timeseries,
        metric_func: Callable[[Timeseries, Timeseries], Unitful | Array],
        is_metric_symmetric: bool = True,
    ) -> Unitful:
        """
        Computes the Continuous Ranked Probability Score (CRPS) for the [`pastax.trajectory.TimeseriesEnsemble`].

        Parameters
        ----------
        other : Timeseries
            The other timeseries to compare against.
        metric_func : Callable[[Timeseries, Timeseries], Unitful | Array]
            The metric function to use.
        is_metric_symmetric : bool, optional
            Whether the metric function is symmetric,
            in which case half of the intra ensemble "distances" are evaluated when computing the ensemble dispersion,
            defaults to `True`.

        Returns
        -------
        Unitful
            The CRPS for the [`pastax.trajectory.TimeseriesEnsemble`].
        """
        biases = self.map(lambda member: metric_func(other, member))
        bias = biases.mean(axis=0)

        n_members = self.size
        dispersion = self.ensemble_dispersion(metric_func, is_metric_symmetric=is_metric_symmetric)
        dispersion /= 2 * n_members * (n_members - 1)

        return bias - dispersion

    def ensemble_dispersion(
        self,
        metric_func: Callable[[Timeseries, Timeseries], Unitful | Array],
        is_metric_symmetric: bool = True,
    ) -> Unitful:
        """
        Computes the [`pastax.trajectory.TimeseriesEnsemble`][] dispersion.

        Parameters
        ----------
        metric_func : Callable[[Timeseries, Timeseries], Unitful | Array]
            The metric function to use.
        is_metric_symmetric : bool, optional
            Whether the metric function is symmetric,
            in which case half of the intra ensemble "distances" are evaluated, defaults to `True`.

        Returns
        -------
        Unitful
            The [`pastax.trajectory.TimeseriesEnsemble`][] dispersion.
        """
        ij = jnp.column_stack(jnp.triu_indices(self.size, k=1))

        if not is_metric_symmetric:
            ij = jnp.vstack([ij, ij[:, ::-1]])

        vmap_metric_fn = eqx.filter_vmap(
            lambda _ij: metric_func(
                self._members_type.from_array(self.value[_ij[0], ...], self.times.value),
                self._members_type.from_array(self.value[_ij[1], ...], self.times.value),
            )
        )

        intra_distances = vmap_metric_fn(ij)
        dispersion = intra_distances.sum(axis=0)

        if is_metric_symmetric:
            dispersion *= 2

        return dispersion

    def map(self, func: Callable[[Timeseries], Unitful | Array]) -> Unitful:
        """
        Applies a function to each [`pastax.trajectory.Timeseries`][] of the [`pastax.trajectory.TimeseriesEnsemble`].

        Parameters
        ----------
        func : Callable[[Timeseries], Unitful | Array]
            The function to apply to each [`pastax.trajectory.Timeseries`][].

        Returns
        -------
        Unitful
            The result of applying the function to each [`pastax.trajectory.Timeseries`][].
        """
        in_axes = eqx.filter(self.members, False)
        in_axes = eqx.tree_at(lambda x: x.states._value, in_axes, 0, is_leaf=lambda x: x is None)
        res = eqx.filter_vmap(func, in_axes=(in_axes,))(self.members)

        unit = {}
        if isinstance(res, Unitful):
            unit = res.unit
            res = res.value

        return Unitful(res, unit)

    def to_xarray(self) -> xr.Dataset:
        """
        Converts the [`pastax.trajectory.TimeseriesEnsemble`][] to a `xarray.Dataset`.

        Returns
        -------
        xr.Dataset
            The corresponding `xarray.Dataset`.
        """
        da = self.to_dataarray()
        ds = da.to_dataset()

        return ds

    @classmethod
    def from_array(
        cls,
        values: Float[Array, "member time state"],
        times: Float[Array, "time"],
        unit: dict[Unit, int | float] = {},
        name: str | None = None,
        **kwargs: Any,
    ) -> TimeseriesEnsemble:
        """
        Creates a [`pastax.trajectory.TimeseriesEnsemble`][] from arrays of values and time points.

        Parameters
        ----------
        values : Float[Array, "member time state"]
            The values for the members of the ensemble.
        times : Float[Array, "time"]
            The time points for the timeseries.
        unit : dict[Unit, int | float], optional
            The unit of the timeseries, defaults to {}.
        name : str | None, optional
            The name of the timeseries, defaults to None.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        TimeseriesEnsemble
            The corresponding [`pastax.trajectory.TimeseriesEnsemble`][].
        """
        members = eqx.filter_vmap(
            lambda x: cls._members_type.from_array(x, times, unit=unit, name=name, **kwargs), out_axes=_if_mapped(0)
        )(values)

        return cls(members)

    def to_dataarray(self) -> xr.DataArray:
        da = xr.DataArray(
            data=self.states.value,
            dims=["member", "time"],
            coords={
                "member": np.arange(self.size),
                "time": self.members.times.to_datetime(),
            },
            name=self.name,
            attrs={"units": units_to_str(self.unit)},
        )

        return da
