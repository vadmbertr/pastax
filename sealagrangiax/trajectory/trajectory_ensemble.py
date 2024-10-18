from __future__ import annotations
from typing import Callable, ClassVar, Dict

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..utils.unit import units_to_str
from ._state import State
from .state import Location
from ._timeseries_ensemble import TimeseriesEnsemble
from .trajectory import Timeseries, Trajectory
from ._unitful import Unitful


class TrajectoryEnsemble(TimeseriesEnsemble):
    """
    Class representing an ensemble of trajectories.

    Attributes
    ----------
    members : Trajectory
        The members of the trajectory ensemble.
    _members_type : ClassVar
        The type of the members in the ensemble (set to Trajectory).

    Methods
    -------
    id
        Returns the IDs of the trajectories.
    latitudes
        Returns the latitudes of the trajectories.
    locations
        Returns the locations of the trajectories.
    longitudes
        Returns the longitudes of the trajectories.
    origin
        Returns the origin of the trajectories.
    crps(other, distance_func=Trajectory.separation_distance)
        Computes the Continuous Ranked Probability Score (CRPS) for the ensemble.
    liu_index(other)
        Computes the Liu Index for each ensemble trajectory.
    lengths()
        Returns the lengths of the trajectories.
    mae(other)
        Computes the Mean Absolute Error (MAE) for each ensemble trajectory.
    plot(ax, label, color, ti=None)
        Plots the trajectories on a given matplotlib axis.
    rmse(other)
        Computes the Root Mean Square Error (RMSE) for each ensemble trajectory.
    separation_distance(other)
        Computes the separation distance for each ensemble trajectory.
    steps()
        Returns the steps of the trajectories.
    from_array(values, times, id=None)
        Creates a TrajectoryEnsemble from an array of values and time points.
    to_dataarray()
        Converts the trajectory ensemble locations to a dict of xarray DataArray.
    to_dataset()
        Converts the trajectory ensemble to a xarray Dataset.
    """

    members: Trajectory
    _members_type: ClassVar = Trajectory

    @property
    def id(self) -> Int[Array, "member"]:
        """
        Returns the IDs of the trajectories.

        Returns
        -------
        Int[Array, "member"]
            The IDs of the trajectories.
        """
        return self.members.id

    @property
    def latitudes(self) -> State:
        """
        Returns the latitudes of the trajectories.

        Returns
        -------
        State
            The latitudes of the trajectories.
        """
        return self.members.latitudes

    @property
    def locations(self) -> Location:
        """
        Returns the locations of the trajectories.

        Returns
        -------
        Location
            The locations of the trajectories.
        """
        return self.members.locations

    @property
    def longitudes(self) -> State:
        """
        Returns the longitudes of the trajectories.

        Returns
        -------
        State
            The longitudes of the trajectories.
        """
        return self.members.longitudes

    @property
    def origin(self) -> State:
        """
        Returns the origin of the trajectories.

        Returns
        -------
        State
            The origin of the trajectories.
        """
        return self.members.origin

    def crps(
        self,
        other: Trajectory,
        distance_func: Callable[[Trajectory, Trajectory], Unitful | ArrayLike] = Trajectory.separation_distance
    ) -> Timeseries:
        """
        Computes the Continuous Ranked Probability Score (CRPS) for the ensemble.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.
        distance_func : Callable[[Trajectory, Trajectory], Unitful | ArrayLike], optional
            The distance function to use (default is Trajectory.separation_distance).

        Returns
        -------
        Timeseries
            The CRPS for the ensemble.
        """
        return super().crps(other, distance_func)

    def liu_index(self, other: Trajectory) -> TimeseriesEnsemble:
        """
        Computes the Liu Index for each ensemble trajectory.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.

        Returns
        -------
        TimeseriesEnsemble
            The Liu Index for each ensemble trajectory.
        """
        return self.map(lambda trajectory: other.liu_index(trajectory))

    def lengths(self) -> TimeseriesEnsemble:
        """
        Returns the lengths of the trajectories.

        Returns
        -------
        TimeseriesEnsemble
            The lengths of the trajectories.
        """
        return self.map(lambda trajectory: trajectory.lengths())

    def mae(self, other: Trajectory) -> TimeseriesEnsemble:
        """
        Computes the Mean Absolute Error (MAE) for each ensemble trajectory.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.

        Returns
        -------
        TimeseriesEnsemble
            The MAE for each ensemble trajectory.
        """
        return self.map(lambda trajectory: other.mae(trajectory))

    def plot(self, ax: plt.Axes, label: str, color: str, ti: int = None) -> plt.Axes:
        """
        Plots the trajectories on a given matplotlib axis.

        Parameters
        ----------
        ax : plt.Axes
            The matplotlib axis to plot on.
        label : str
            The label for the plot.
        color : str
            The color for the plot.
        ti : int, optional
            The time index to plot up to (default is None).

        Returns
        -------
        plt.Axes
            The matplotlib axis with the plot.
        """
        if ti is None:
            ti = self.length

        alpha_factor = jnp.clip(1 / ((self.size / 10) ** 0.5), .05, 1).item()
        alpha = jnp.geomspace(.25, 1, ti) * alpha_factor

        locations = self.locations.value.swapaxes(0, 1)[:ti, :, None, ::-1]
        segments = jnp.concat([locations[:-1], locations[1:]], axis=2).reshape(-1, 2, 2)
        alpha = jnp.repeat(alpha, self.size)

        lc = LineCollection(segments, color=color, alpha=alpha)
        ax.add_collection(lc)

        lc = LineCollection(segments[-self.size:, ...], color=color, alpha=alpha_factor)
        ax.add_collection(lc)

        ax.plot(self.longitudes.value[0, -1], self.latitudes.value[0, -1], label=label, color=color)  # for label display

        return ax

    def rmse(self, other: Trajectory) -> TimeseriesEnsemble:
        """
        Computes the Root Mean Square Error (RMSE) for each ensemble trajectory.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.

        Returns
        -------
        TimeseriesEnsemble
            The RMSE for each ensemble trajectory.
        """
        return self.map(lambda trajectory: other.rmse(trajectory))

    def separation_distance(self, other: Trajectory) -> TimeseriesEnsemble:
        """
        Computes the separation distance for each ensemble trajectory.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.

        Returns
        -------
        TimeseriesEnsemble
            The separation distance for each ensemble trajectory.
        """
        return self.map(lambda trajectory: other.separation_distance(trajectory))

    def steps(self) -> TimeseriesEnsemble:
        """
        Returns the steps of the trajectories.

        Returns
        -------
        TimeseriesEnsemble
            The steps of the trajectories.
        """
        return self.map(lambda trajectory: trajectory.steps())

    @classmethod
    def from_array(
        cls,
        values: Float[Array, "member time 2"],
        times: Float[Array, "time"],
        id: Int[Array, ""] = None
    ) -> TrajectoryEnsemble:
        """
        Creates a TrajectoryEnsemble from an array of values and time points.

        Parameters
        ----------
        values : Float[Array, "member time 2"]
            The values for the members of the trajectory ensemble.
        times : Float[Array, "time"]
            The time points for the timeseries.
        id : Int[Array, ""], optional
            The ID of the trajectory (default is None).

        Returns
        -------
        TrajectoryEnsemble
            The TrajectoryEnsemble created from the array of values and time points.
        """
        return super().from_array(values, times, id=id)

    def to_dataarray(self) -> Dict[str, xr.DataArray]:
        """
        Converts the ensemble states to a dictionary of xarray DataArrays.

        Returns
        -------
        Dict[str, xr.DataArray]
            A dictionary where keys are the variable names and values are the corresponding xarray DataArrays.
        """
        member = np.arange(self.size)
        times = self.members.times.to_datetime()
        unit = units_to_str(self.unit)

        id_da = xr.DataArray(
            data=self.id,
            dims=["member"],
            coords={"member": member},
            name="drifter id"
        )
        latitude_da = xr.DataArray(
            data=self.latitudes,
            dims=["member", "time"],
            coords={"member": member, "time": times},
            name="latitude",
            attrs={"units": unit}
        )
        longitude_da = xr.DataArray(
            data=self.longitudes,
            dims=["member", "time"],
            coords={"member": member, "time": times},
            name="longitude",
            attrs={"units": unit}
        )

        return {"id": id_da, "latitude": latitude_da, "longitude": longitude_da}

    def to_dataset(self) -> xr.Dataset:
        """
        Converts the ensemble states to an xarray Dataset.

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the ensemble states.
        """
        return xr.Dataset(self.to_dataarray())
