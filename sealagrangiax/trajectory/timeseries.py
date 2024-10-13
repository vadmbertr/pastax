from __future__ import annotations
from typing import ClassVar, Dict

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import xarray as xr

from ..utils import UNIT
from ..utils.geo import earth_distance
from ._timeseries import Timeseries
from ._state import WHAT
from .state import Location


class Trajectory(Timeseries):
    """
    Class representing a trajectory with 2D geographical locations over time.

    Attributes
    ----------
    _states : Location
        The locations of the trajectory.
    _states_type : ClassVar
        The type of the states in the trajectory (set to Location).
    id : Int[Array, ""], optional
        The ID of the trajectory (defaults is None).

    Methods
    -------
    __init__(locations, times, trajectory_id=None, **_)
        Initializes the Trajectory with given locations, times, and optional trajectory ID.
    latitudes
        Returns the latitudes of the trajectory.
    locations
        Returns the locations of the trajectory.
    longitudes
        Returns the longitudes of the trajectory.
    origin
        Returns the origin of the trajectory.
    _locations
        Returns the locations of the trajectory as Location objects.
    lengths
        Returns the cumulative lengths of the trajectory.
    liu_index(other)
        Computes the Liu Index for the trajectory.
    mae(other)
        Computes the Mean Absolute Error (MAE) for the trajectory.
    plot(ax, label, color, ti=None)
        Plots the trajectory on a given matplotlib axis.
    rmse(other)
        Computes the Root Mean Square Error (RMSE) for the trajectory.
    separation_distance(other)
        Computes the separation distance between this trajectory and another trajectory.
    steps
        Returns the steps of the trajectory.
    """

    _states: Location
    _states_type: ClassVar = Location
    id: Int[Array, ""] = None

    def __init__(
        self, locations: Float[Array, "time 2"], times: Int[Array, "time"], trajectory_id: Int[Array, ""] = None, **_
    ):
        """
        Initializes the Trajectory with given locations, times, and optional trajectory ID.

        Parameters
        ----------
        locations : Float[Array, "time 2"]
            The locations for the trajectory.
        times : Int[Array, "time"]
            The time points for the trajectory.
        trajectory_id : Int[Array, ""], optional
            The ID of the trajectory (default is None).
        **_
            Additional keyword arguments.
        """
        super().__init__(locations, times, what=WHAT.location, unit=UNIT.degrees)
        self.id = trajectory_id

    @property
    @eqx.filter_jit
    def latitudes(self) -> Float[Array, "time"]:
        """
        Returns the latitudes of the trajectory.

        Returns
        -------
        Float[Array, "time"]
            The latitudes of the trajectory.
        """
        return self.locations[:, 0]

    @property
    @eqx.filter_jit
    def locations(self) -> Float[Array, "time 2"]:
        """
        Returns the locations of the trajectory.

        Returns
        -------
        Float[Array, "time 2"]
            The locations of the trajectory.
        """
        return self.states

    @property
    @eqx.filter_jit
    def longitudes(self) -> Float[Array, "time"]:
        """
        Returns the longitudes of the trajectory.

        Returns
        -------
        Float[Array, "time"]
            The longitudes of the trajectory.
        """
        return self.locations[:, 1]

    @property
    @eqx.filter_jit
    def origin(self) -> Float[Array, "2"]:
        """
        Returns the origin of the trajectory.

        Returns
        -------
        Float[Array, "2"]
            The origin of the trajectory.
        """
        return self.locations[0]

    @property
    @eqx.filter_jit
    def _locations(self) -> Location:
        """
        Returns the locations of the trajectory as Location objects.

        Returns
        -------
        Location
            The locations of the trajectory as Location objects.
        """
        return self._states

    @eqx.filter_jit
    def lengths(self) -> Float[Array, "time"]:
        """
        Returns the cumulative lengths of the trajectory.

        Returns
        -------
        Float[Array, "time"]
            The cumulative lengths of the trajectory.
        """
        return jnp.cumsum(self.steps())

    @eqx.filter_jit
    def liu_index(self, other: Trajectory) -> Float[Array, "time"]:
        """
        Computes the Liu Index (over time) between this trajectory and another trajectory.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.

        Returns
        -------
        Float[Array, "time"]
            The Liu Index between the two trajectories.
        """
        error = self.separation_distance(other).cumsum()
        cum_lengths = self.lengths().cumsum()
        liu_index = error / cum_lengths

        return liu_index

    @eqx.filter_jit
    def mae(self, other: Trajectory) -> Float[Array, "time"]:
        """
        Computes the Mean Absolute Error (MAE) (over time) between this trajectory and another trajectory.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.

        Returns
        -------
        Float[Array, "time"]
            The MAE between the two trajectories.
        """
        error = self.separation_distance(other).cumsum()
        length = jnp.arange(self.length)  # we consider that traj starts from the same x0
        mae = error / length

        return mae

    def plot(self, ax: plt.Axes, label: str, color: str, ti: int = None) -> plt.Axes:
        """
        Plots the trajectory on a given matplotlib axis.

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

        alpha = jnp.geomspace(.25, 1, ti)

        locations = self.locations[:ti, None, ::-1]
        segments = jnp.concat([locations[:-1], locations[1:]], axis=1)

        lc = LineCollection(segments)
        lc.set_color(color)
        lc.set_alpha(alpha)
        ax.add_collection(lc)

        ax.plot(locations[-2:, 0, 0], locations[-2:, 0, 1], label=label, color=color)

        return ax

    @eqx.filter_jit
    def rmse(self, other: Trajectory) -> Float[Array, "time"]:
        """
        Computes the Root Mean Square Error (RMSE) (over time) between this trajectory and another trajectory.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.

        Returns
        -------
        Float[Array, "time"]
            The RMSE between the two trajectories.
        """
        error = (self.separation_distance(other) ** 2).cumsum()
        length = jnp.arange(self.length)  # we consider that traj starts from the same x0
        rmse = (error / length) ** (1 / 2)

        return rmse

    @eqx.filter_jit
    def separation_distance(self, other: Trajectory) -> Float[Array, "time"]:
        """
        Computes the separation distance (over time) between this trajectory and another trajectory.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.

        Returns
        -------
        Float[Array, "time"]
            The separation distance between the two trajectories.
        """
        def axes_func(leaf):
            axes = None
            if eqx.is_array(leaf) and leaf.ndim > 0:
                axes = 0
            return axes

        separation_distance = eqx.filter_vmap(lambda p1, p2: p1.earth_distance(p2), in_axes=axes_func)(
            self._locations, other._locations
        )

        return separation_distance

    @eqx.filter_jit
    def steps(self) -> Float[Array, "time"]:
        """
        Returns the steps of the trajectory.

        Returns
        -------
        Float[Array, "time"]
            The steps of the trajectory.
        """
        def axes_func(leaf):
            axes = None
            if eqx.is_array(leaf) and leaf.ndim > 0:
                axes = 0
            return axes

        steps = eqx.filter_vmap(lambda p1, p2: earth_distance(p1, p2), in_axes=axes_func)(
            self.locations[1:], self.locations[:-1]
        )
        steps = jnp.pad(steps, (1, 0), constant_values=0.)  # adds a 1st 0 step

        return steps

    def to_dataarray(self) -> Dict[str, xr.DataArray]:
        """
        Converts the timeseries states to a dictionary of xarray DataArrays.

        Returns
        -------
        Dict[str, xr.DataArray]
            A dictionary where keys are the variable names and values are the corresponding xarray DataArrays.
        """
        times = self._times.to_datetime()
        unit = UNIT[self.unit]

        latitude_da = xr.DataArray(
            data=self.latitudes,
            dims=["time"],
            coords={"time": times},
            name="latitude",
            attrs={"units": unit}
        )
        longitude_da = xr.DataArray(
            data=self.longitudes,
            dims=["time"],
            coords={"time": times},
            name="longitude",
            attrs={"units": unit}
        )

        return {"latitude": latitude_da, "longitude": longitude_da}

    def to_dataset(self) -> xr.Dataset:
        """
        Converts the timeseries states to an xarray Dataset.

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the time seriesstates.
        """
        return xr.Dataset(self.to_dataarray())
