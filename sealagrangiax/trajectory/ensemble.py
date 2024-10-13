from typing import Callable, ClassVar, Dict

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..utils import UNIT
from ._ensemble import TimeseriesEnsemble
from ._state import WHAT
from .timeseries import Timeseries, Trajectory


class TrajectoryEnsemble(TimeseriesEnsemble):
    """
    Class representing an ensemble of trajectories.

    Attributes
    ----------
    _members : Trajectory
        The members of the trajectory ensemble.
    _members_type : ClassVar
        The type of the members in the ensemble (set to Trajectory).

    Methods
    -------
    __init__(locations, times, **_)
        Initializes the TrajectoryEnsemble with given locations and times.
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
    lengths
        Returns the lengths of the trajectories.
    mae(other)
        Computes the Mean Absolute Error (MAE) for each ensemble trajectory.
    plot(ax, label, color, ti=None)
        Plots the trajectories on a given matplotlib axis.
    rmse(other)
        Computes the Root Mean Square Error (RMSE) for each ensemble trajectory.
    separation_distance(other)
        Computes the separation distance for each ensemble trajectory.
    steps
        Returns the steps of the trajectories.
    """

    _members: Trajectory
    _members_type: ClassVar = Trajectory

    def __init__(
            self,
            locations: Float[Array, "member time 2"],
            times: Int[Array, "time"],
            **_
    ):
        """
        Initializes the TrajectoryEnsemble with given locations and times.

        Parameters
        ----------
        locations : Float[Array, "member time 2"]
            The locations for the members of the ensemble.
        times : Int[Array, "time"]
            The time points for the trajectories.
        **_
            Additional keyword arguments.
        """
        super().__init__(locations, times, what=WHAT.location, unit=UNIT.degrees)

    @property
    @eqx.filter_jit
    def id(self) -> Int[Array, "member"]:
        """
        Returns the IDs of the trajectories.

        Returns
        -------
        Int[Array, "member"]
            The IDs of the trajectories.
        """
        return self._members.id

    @property
    @eqx.filter_jit
    def latitudes(self) -> Float[Array, "member time"]:
        """
        Returns the latitudes of the trajectories.

        Returns
        -------
        Float[Array, "member time"]
            The latitudes of the trajectories.
        """
        return self.states[..., 0]

    @property
    @eqx.filter_jit
    def locations(self) -> Float[Array, "member time 2"]:
        """
        Returns the locations of the trajectories.

        Returns
        -------
        Float[Array, "member time 2"]
            The locations of the trajectories.
        """
        return self.states

    @property
    @eqx.filter_jit
    def longitudes(self) -> Float[Array, "member time"]:
        """
        Returns the longitudes of the trajectories.

        Returns
        -------
        Float[Array, "member time"]
            The longitudes of the trajectories.
        """
        return self.states[..., 1]

    @property
    @eqx.filter_jit
    def origin(self) -> Float[Array, "2"]:
        """
        Returns the origin of the trajectories.

        Returns
        -------
        Float[Array, "2"]
            The origin of the trajectories.
        """
        return self.locations[..., 0, 0]

    @eqx.filter_jit
    def crps(
            self,
            other: Trajectory,
            distance_func: Callable[[Trajectory, Trajectory], Float[Array, "time"]] = Trajectory.separation_distance
    ) -> Float[Array, "time"] | Timeseries:
        """
        Computes the Continuous Ranked Probability Score (CRPS) for the ensemble.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.
        distance_func : Callable[[Trajectory, Trajectory], Float[Array, "time"]], optional
            The distance function to use (default is Trajectory.separation_distance).

        Returns
        -------
        Float[Array, "time"] | Timeseries
            The CRPS for the ensemble.
        """
        return super().crps(other, distance_func)

    @eqx.filter_jit
    def liu_index(self, other: Trajectory) -> Float[Array, "member time"]:
        """
        Computes the Liu Index for each ensemble trajectory.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.

        Returns
        -------
        Float[Array, "member time"]
            The Liu Index for each ensemble trajectory.
        """
        return self.map(lambda trajectory: other.liu_index(trajectory))

    @eqx.filter_jit
    def lengths(self) -> Float[Array, "member time"]:
        """
        Returns the lengths of the trajectories.

        Returns
        -------
        Float[Array, "member time"]
            The lengths of the trajectories.
        """
        return self.map(lambda trajectory: trajectory.lengths())

    @eqx.filter_jit
    def mae(self, other: Trajectory) -> Float[Array, "member time"]:
        """
        Computes the Mean Absolute Error (MAE) for each ensemble trajectory.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.

        Returns
        -------
        Float[Array, "member time"]
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

        locations = self.locations.swapaxes(0, 1)[:ti, :, None, ::-1]
        segments = jnp.concat([locations[:-1], locations[1:]], axis=2).reshape(-1, 2, 2)
        alpha = jnp.repeat(alpha, self.size)

        lc = LineCollection(segments, color=color, alpha=alpha)
        ax.add_collection(lc)

        lc = LineCollection(segments[-self.size:, ...], color=color, alpha=alpha_factor)
        ax.add_collection(lc)

        ax.plot(self.longitudes[0, -1], self.latitudes[0, -1], label=label, color=color)  # for label display

        return ax

    @eqx.filter_jit
    def rmse(self, other: Trajectory) -> Float[Array, "member time"]:
        """
        Computes the Root Mean Square Error (RMSE) for each ensemble trajectory.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.

        Returns
        -------
        Float[Array, "member time"]
            The RMSE for each ensemble trajectory.
        """
        return self.map(lambda trajectory: other.rmse(trajectory))

    @eqx.filter_jit
    def separation_distance(self, other: Trajectory) -> Float[Array, "member time"]:
        """
        Computes the separation distance for each ensemble trajectory.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.

        Returns
        -------
        Float[Array, "member time"]
            The separation distance for each ensemble trajectory.
        """
        return self.map(lambda trajectory: other.separation_distance(trajectory))

    @eqx.filter_jit
    def steps(self) -> Float[Array, "member time"]:
        """
        Returns the steps of the trajectories.

        Returns
        -------
        Float[Array, "member time"]
            The steps of the trajectories.
        """
        return self.map(lambda trajectory: trajectory.steps())

    def to_dataarray(self) -> Dict[str, xr.DataArray]:
        """
        Converts the ensemble states to a dictionary of xarray DataArrays.

        Returns
        -------
        Dict[str, xr.DataArray]
            A dictionary where keys are the variable names and values are the corresponding xarray DataArrays.
        """
        member = np.arange(self.size)
        times = self._members._times.to_datetime()
        unit = UNIT[self.unit]

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
