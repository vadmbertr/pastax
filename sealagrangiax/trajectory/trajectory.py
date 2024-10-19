from __future__ import annotations
from typing import ClassVar, Dict

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import xarray as xr

from ..utils.geo import distance_on_earth
from ..utils.unit import UNIT, units_to_str
from ._state import State
from .state import Location, Time
from ._timeseries import _in_axes_func, Timeseries


class Trajectory(Timeseries):
    """
    Class representing a trajectory with 2D geographical locations over time.

    Attributes
    ----------
    states : Location
        The locations of the trajectory.
    _states_type : ClassVar
        The type of the states in the trajectory (set to Location).
    id : Int[Array, ""], optional
        The ID of the trajectory (defaults is None).

    Methods
    -------
    __init__(locations, times, id=None, **_)
        Initializes the Trajectory with given locations, times, and optional trajectory ID.
    latitudes
        Returns the latitudes of the trajectory.
    locations
        Returns the locations of the trajectory.
    longitudes
        Returns the longitudes of the trajectory.
    origin
        Returns the origin of the trajectory.
    lengths()
        Returns the cumulative lengths of the trajectory.
    liu_index(other)
        Computes the Liu Index between this trajectory and another trajectory.
    mae(other)
        Computes the Mean Absolute Error (MAE) between this trajectory and another trajectory.
    plot(ax, label, color, ti=None)
        Plots the trajectory on a given matplotlib axis.
    rmse(other)
        Computes the Root Mean Square Error (RMSE) between this trajectory and another trajectory.
    separation_distance(other)
        Computes the separation distance between this trajectory and another trajectory.
    steps()
        Returns the steps of the trajectory.
    from_array(values, times, id=None)
        Creates a Trajectory from an array of values and time points.
    to_dataarray()
        Converts the trajectory locations to a dict of xarray DataArray.
    to_dataset()
        Converts the trajectory to a xarray Dataset.
    """

    states: Location
    _states_type: ClassVar = Location
    id: Int[Array, ""] = None

    def __init__(
        self,
        locations: Location,
        times: Time,
        id: Int[Array, ""] = None,
        *_,
        **__
    ):
        """
        Initializes the Trajectory with given locations, times, and optional trajectory ID.

        Parameters
        ----------
        locations : Float[Array, "... time 2"]
            The locations for the trajectory.
        times : Int[Array, "... time"]
            The time points for the trajectory.
        """
        super().__init__(locations, times)
        self.id = id

    @property
    def latitudes(self) -> State:
        """
        Returns the latitudes of the trajectory.

        Returns
        -------
        State
            The latitudes of the trajectory.
        """
        return self.locations.latitude

    @property
    def locations(self) -> Location:
        """
        Returns the locations of the trajectory.

        Returns
        -------
        Location
            The locations of the trajectory.
        """
        return self.states

    @property
    def longitudes(self) -> State:
        """
        Returns the longitudes of the trajectory.

        Returns
        -------
        State
            The longitudes of the trajectory.
        """
        return self.locations.longitude

    @property
    def origin(self) -> State:
        """
        Returns the origin of the trajectory.

        Returns
        -------
        State
            The origin of the trajectory.
        """
        return State(self.locations.value[..., 0, :], unit=self.unit, name="Origin in [latitude, longitude]")

    def lengths(self) -> Timeseries:
        """
        Returns the cumulative lengths of the trajectory.

        Returns
        -------
        Timeseries
            The cumulative lengths of the trajectory.
        """
        lengths = self.steps().cumsum()
        return Timeseries.from_array(lengths.value, self.times.value, unit=lengths.unit, name="Cumulative lengths")

    def liu_index(self, other: Trajectory) -> Timeseries:
        """
        Computes the Liu Index (over time) between this trajectory and another trajectory.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.

        Returns
        -------
        Timeseries
            The Liu Index between the two trajectories.
        """
        error = self.separation_distance(other).cumsum()
        cum_lengths = self.lengths().cumsum()
        liu_index = error / cum_lengths

        return Timeseries.from_array(liu_index.value, self.times.value, name="Liu index")

    def mae(self, other: Trajectory) -> Timeseries:
        """
        Computes the Mean Absolute Error (MAE) (over time) between this trajectory and another trajectory.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.

        Returns
        -------
        Timeseries
            The MAE between the two trajectories.
        """
        error = self.separation_distance(other).cumsum()
        length = jnp.arange(self.length)  # we consider that traj starts from the same x0
        mae = error / length

        return Timeseries.from_array(mae.value, self.times.value, mae.unit, name="MAE")

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

        locations = self.locations.value[:ti, None, ::-1]
        segments = jnp.concat([locations[:-1], locations[1:]], axis=1)

        lc = LineCollection(segments)
        lc.set_color(color)
        lc.set_alpha(alpha)
        ax.add_collection(lc)

        ax.plot(locations[-2:, 0, 0], locations[-2:, 0, 1], label=label, color=color)

        return ax

    def rmse(self, other: Trajectory) -> Timeseries:
        """
        Computes the Root Mean Square Error (RMSE) (over time) between this trajectory and another trajectory.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.

        Returns
        -------
        Timeseries
            The RMSE between the two trajectories.
        """
        error = (self.separation_distance(other) ** 2).cumsum()
        length = jnp.arange(self.length)  # we consider that traj starts from the same x0
        rmse = (error / length) ** (1 / 2)

        return Timeseries.from_array(rmse.value, self.times.value, rmse.unit, name="RMSE")

    def separation_distance(self, other: Trajectory) -> Timeseries:
        """
        Computes the separation distance (over time) between this trajectory and another trajectory.

        Parameters
        ----------
        other : Trajectory
            The other trajectory to compare against.

        Returns
        -------
        Timeseries
            The separation distance between the two trajectories.
        """
        separation_distance = eqx.filter_vmap(lambda p1, p2: p1.distance_on_earth(p2), in_axes=_in_axes_func)(
            self.locations, other.locations
        )

        return Timeseries.from_array(
            separation_distance.value, 
            self.times.value, 
            separation_distance.unit, 
            name="Separation distance"
        )

    def steps(self) -> Timeseries:
        """
        Returns the steps of the trajectory.

        Returns
        -------
        Timeseries
            The steps of the trajectory.
        """
        steps = eqx.filter_vmap(lambda p1, p2: distance_on_earth(p1, p2), in_axes=_in_axes_func)(
            self.locations.value[1:], self.locations.value[:-1]
        )

        steps = jnp.pad(steps, (1, 0), constant_values=0.)  # adds a 1st 0 step

        return Timeseries.from_array(steps, self.times.value, UNIT["m"], name="Trajectory steps")
    
    @classmethod
    def from_array(
        cls, 
        values: Float[Array, "... time 2"], 
        times: Float[Array, "... time"],
        id: Int[Array, ""] = None,
        **_: Dict
    ) -> Trajectory:
        """
        Creates a trajectory from an array of values and time points.

        Parameters
        ----------
        values : Float[Array, "... time 2"]
            The array of values for the trajectory.
        times : Float[Array, "... time"]
            The time points for the trajectory.
        id : Int[Array, ""], optional
            The ID of the trajectory (default is None).

        Returns
        -------
        Trajectory
            The trajectory created from the array of values and time points.
        """
        return super().from_array(values, times, unit=UNIT["°"], name="Location in [latitude, longitude]", id=id)

    def to_dataarray(self) -> Dict[str, xr.DataArray]:
        """
        Converts the trajectory location to a dictionary of xarray DataArrays.

        Returns
        -------
        Dict[str, xr.DataArray]
            A dictionary where keys are the variable names and values are the corresponding xarray DataArrays.
        """
        times = self.times.to_datetime()
        unit = units_to_str(self.unit)

        latitude_da = xr.DataArray(
            data=self.latitudes.value,
            dims=["time"],
            coords={"time": times},
            name="latitude",
            attrs={"units": unit}
        )
        longitude_da = xr.DataArray(
            data=self.longitudes.value,
            dims=["time"],
            coords={"time": times},
            name="longitude",
            attrs={"units": unit}
        )

        return {"latitude": latitude_da, "longitude": longitude_da}

    def to_dataset(self) -> xr.Dataset:
        """
        Converts the trajectory to an xarray Dataset.

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the time seriesstates.
        """
        return xr.Dataset(self.to_dataarray())
