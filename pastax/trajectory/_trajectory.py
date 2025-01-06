from __future__ import annotations

from typing import Any, ClassVar

import cartopy.crs as ccrs
import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import xarray as xr
from jaxtyping import Array, Float, Int
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from ..utils._geo import distance_on_earth
from ..utils._unit import time_in_seconds, UNIT, Unit, units_to_str
from ._state import State
from ._states import Location, Time
from ._timeseries import Timeseries


class Trajectory(Timeseries):
    """
    Class representing a trajectory with 2D geographical locations over time.

    Attributes
    ----------
    states : Location
        The locations of the trajectory.
    id : Int[Array, ""] | None, optional
        The ID of the trajectory, defaults to `None`.

    Methods
    -------
    __init__(locations, times, id=None, **_)
        Initializes the [`pastax.trajectory.Trajectory`][] with given locations, times, and optional trajectory ID.
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
    plot(ax=None, label=None, color=None, alpha_factor=1, ti=None)
        Plots the trajectory.
    rmse(other)
        Computes the Root Mean Square Error (RMSE) between this trajectory and another trajectory.
    separation_distance(other)
        Computes the separation distance between this trajectory and another trajectory.
    steps()
        Returns the steps of the trajectory.
    to_xarray()
        Converts the [`pastax.trajectory.Trajectory`][] to a `xarray.Dataset`.
    from_array(values, times, unit=UNIT["°"], id=None)
        Creates a [`pastax.trajectory.Trajectory`][] from arrays of (latitudes, longitudes) values and time points.
    from_xarray(dataset, time_varname="time", lat_varname="lat", lon_varname="lon", unit=UNIT["°"], id=None)
        Creates a [`pastax.trajectory.Trajectory`][] from a `xarray.Dataset`.
    """

    states: Location
    _states_type: ClassVar = Location
    id: Int[Array, ""] | None = None

    def __init__(
        self,
        locations: Location,
        times: Time,
        id: Int[Array, ""] | None = None,
        *_,
        **__,
    ):
        """
        Initializes the Trajectory with given locations, times, and optional trajectory ID.

        Parameters
        ----------
        locations : Float[Array, "... time 2"]
            The locations for the trajectory.
        times : Int[Array, "... time"]
            The time points for the trajectory.
        id : Int[Array, ""] | None, optional
            The ID of the trajectory, defaults to None.
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
        return State(
            self.locations.value[..., 0, :],
            unit=self.unit,
            name="Origin in [latitude, longitude]",
        )

    def lengths(self) -> Timeseries:
        """
        Returns the cumulative lengths of the trajectory.

        Returns
        -------
        Timeseries
            The cumulative lengths of the trajectory.
        """
        lengths = self.steps().cumsum()
        return Timeseries.from_array(
            lengths.value,
            self.times.value,
            unit=lengths.unit,
            name="Cumulative lengths",
        )

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
        error = self.separation_distance(other).value.cumsum()
        cum_lengths = self.lengths().value.cumsum()
        cum_lengths = jnp.where(cum_lengths == 0, jnp.inf, cum_lengths)
        liu_index = error / cum_lengths

        return Timeseries.from_array(liu_index, self.times.value, name="Liu index")

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
        length = jnp.arange(self.length) + 1
        mae = error / length

        return Timeseries.from_array(mae.value, self.times.value, mae.unit, name="MAE")

    def plot(
        self,
        ax: Axes | None = None,
        label: str | None = None,
        color: str | None = None,
        alpha_factor: float = 1,
        ti: int | None = None,
        **kwargs,
    ) -> Axes:
        """
        Plots the trajectory.

        Parameters
        ----------
        ax : Axes | None, optional
            The matplotlib axis to plot on, defaults to `None`.
            If `None`, a new figure and axis are created.
        label : str | None, optional
            The label for the plot, defaults to `None`.
        color : str | None, optional
            The color for the plot, defaults to `None`.
        alpha_factor : float, optional
            A factor controlling the overall transparency of the plotted trajectory, defaults to `1`.
        ti : int | None, optional
            The time index to plot up to, defaults to `None`.
        kwargs: dict, optional
            Additional arguments passed to `LineCollection`.

        Returns
        -------
        Axes
            The matplotlib axis with the plot.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection=ccrs.PlateCarree())

        if ti is None:
            ti = self.length

        alpha = jnp.geomspace(0.25, 1, ti - 1) * alpha_factor

        locations = self.locations.value[:ti, None, ::-1]
        segments = jnp.concat([locations[:-1], locations[1:]], axis=1)

        lc = LineCollection(segments, color=color, alpha=alpha, **kwargs)  # type: ignore
        ax.add_collection(lc)

        # trick to display label with alpha=1
        ax.plot(
            self.longitudes.value[-1],
            self.latitudes.value[-1],
            label=label,
            color=color,
        )

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
        length = jnp.arange(self.length) + 1
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
        separation_distance = eqx.filter_vmap(lambda p1, p2: p1.distance_on_earth(p2))(self.locations, other.locations)

        return Timeseries.from_array(
            separation_distance.value,
            self.times.value,
            separation_distance.unit,
            name="Separation distance",
        )

    def steps(self) -> Timeseries:
        """
        Returns the steps of the trajectory.

        Returns
        -------
        Timeseries
            The steps of the trajectory.
        """
        steps = eqx.filter_vmap(lambda p1, p2: distance_on_earth(p1, p2))(
            self.locations.value[1:], self.locations.value[:-1]
        )

        steps = jnp.pad(steps, (1, 0), constant_values=0.0)  # adds a 1st 0 step

        return Timeseries.from_array(steps, self.times.value, UNIT["m"], name="Trajectory steps")

    def to_xarray(self) -> xr.Dataset:
        """
        Converts the [`pastax.trajectory.Trajectory`][] to a `xarray.Dataset`.

        Returns
        -------
        xr.Dataset
            The corresponding `xarray.Dataset`.
        """
        return xr.Dataset(self.to_dataarray())

    @classmethod
    def from_array(
        cls,
        values: Float[Array, "... time 2"],
        times: Float[Array, "... time"],
        unit: dict[Unit, int | float] = UNIT["°"],
        id: Int[Array, ""] | None = None,
        **_: Any,
    ) -> Trajectory:
        """
        Creates a [`pastax.trajectory.Trajectory`][] from arrays of (latitudes, longitudes) values and time points.

        Parameters
        ----------
        values : Float[Array, "... time 2"]
            The array of (latitudes, longitudes) values for the trajectory.
        times : Float[Array, "... time"]
            The time points for the trajectory.
        unit : dict[Unit, int | float], optional
            Unit of the trajectory locations, defaults to UNIT["°"].
        id : Int[Array, ""] | None, optional
            The ID of the trajectory, defaults to None.

        Returns
        -------
        Trajectory
            The corresponding [`pastax.trajectory.Trajectory`][].
        """
        return super().from_array(values, times, unit=unit, name="Location in [latitude, longitude]", id=id)  # type: ignore

    @classmethod
    def from_xarray(
        cls,
        dataset: xr.Dataset,
        time_varname: str = "time",
        lat_varname: str = "lat",  # follows clouddrift "convention"
        lon_varname: str = "lon",  # follows clouddrift "convention"
        unit: dict[Unit, int | float] = UNIT["°"],
        id: Int[Array, ""] | None = None,
        **_: Any,
    ) -> Trajectory:
        """
        Creates a [`pastax.trajectory.Trajectory`][] from a `xarray.Dataset`.

        Parameters
        ----------
        dataset : xr.Dataset
            The `xarray.Dataset` containing the trajectory data.
        time_varname : str, optional
            A string indicating the name of the time variable in the dataset, defaults to `time`.
        lat_varname : str, optional
            A string indicating the name of the latitude variable in the dataset, defaults to `lat`.
        lon_varname : str, optional
            A string indicating the name of the longitude variable in the dataset, defaults to `lon`.
        unit : dict[Unit, int | float], optional
            Unit of the trajectory locations, defaults to UNIT["°"].
        id : Int[Array, ""] | None, optional
            The ID of the trajectory, defaults to None.

        Returns
        -------
        Trajectory
            The corresponding [`pastax.trajectory.Trajectory`][].
        """
        values = jnp.stack([dataset[lat_varname].values, dataset[lon_varname].values], axis=-1)
        times: Array = time_in_seconds(dataset[time_varname].values)
        return cls.from_array(values, times, unit=unit, id=id)

    def to_dataarray(self) -> dict[str, xr.DataArray]:
        times = self.times.to_datetime()
        unit = units_to_str(self.unit)

        latitude_da = xr.DataArray(
            data=self.latitudes.value,
            dims=["obs"],
            coords={"time": ("obs", times)},
            name="lat",
            attrs={"units": unit},
        )
        longitude_da = xr.DataArray(
            data=self.longitudes.value,
            dims=["obs"],
            coords={"time": ("obs", times)},
            name="lon",
            attrs={"units": unit},
        )

        return {"lat": latitude_da, "lon": longitude_da}
