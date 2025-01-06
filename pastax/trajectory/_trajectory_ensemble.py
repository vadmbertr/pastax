from __future__ import annotations

from typing import ClassVar

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from jaxtyping import Array, Float, Int
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from ..utils._unit import UNIT, Unit, units_to_str
from ._state import State
from ._states import Location
from ._timeseries_ensemble import TimeseriesEnsemble
from ._trajectory import Trajectory


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
    plot(ax=None, label=None, color=None, alpha_factor=1, ti=None)
        Plots the trajectories.
    rmse(other)
        Computes the Root Mean Square Error (RMSE) for each ensemble trajectory.
    separation_distance(other)
        Computes the separation distance for each ensemble trajectory.
    steps()
        Returns the steps of the trajectories.
    to_xarray()
        Converts the [`pastax.trajectory.TrajectoryEnsemble`][] to a `xarray.Dataset`.
    from_array(values, times, unit=UNIT["°"]
        Creates a [`pastax.trajectory.TrajectoryEnsemble`][] from arrays of values and time points.
    """

    members: Trajectory
    _members_type: ClassVar = Trajectory

    @property
    def id(self) -> Int[Array, "member"] | None:
        """
        Returns the IDs of the trajectories.

        Returns
        -------
        Int[Array, "member"] | None
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
        liu_index = self.map(lambda trajectory: other.liu_index(trajectory))  # type: ignore
        return TimeseriesEnsemble.from_array(liu_index.value, self.times.value, name="Liu index")

    def lengths(self) -> TimeseriesEnsemble:
        """
        Returns the lengths of the trajectories.

        Returns
        -------
        TimeseriesEnsemble
            The lengths of the trajectories.
        """
        lengths = self.map(lambda trajectory: trajectory.lengths())  # type: ignore
        return TimeseriesEnsemble.from_array(lengths.value, self.times.value, unit=lengths.unit, name="lengths")

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
        mae = self.map(lambda trajectory: other.mae(trajectory))  # type: ignore
        return TimeseriesEnsemble.from_array(mae.value, self.times.value, unit=mae.unit, name="MAE")

    def plot(
        self,
        ax: Axes | None = None,
        label: str | list[str] | None = None,
        color: str | list[str | float | int] | None = None,
        alpha_factor: float = 1,
        ti: int | None = None,
        **kwargs,
    ) -> Axes:
        """
        Plots the trajectories.

        Parameters
        ----------
        ax : Axes | None, optional
            The matplotlib axis to plot on, defaults to `None`.
        label : str | list[str] | None, optional
            The label(s) for the plot, defaults to `None`.
        color : str | list[str | float | int] | None, optional
            The color(s) for the plot, defaults to `None`.
        alpha_factor : float, optional
            A factor controlling the overall transparency of the plotted ensemble, defaults to `1`.
        ti : int | None, optional
            The time index to plot up to, defaults to None.
        kwargs: dict, optional
            Additional arguments passed to `LineCollection`.

        Returns
        -------
        plt.Axes
            The matplotlib axis with the plot.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection=ccrs.PlateCarree())

        if ti is None:
            ti = self.length

        alpha_factor *= np.clip(1 / ((self.size / 10) ** 0.5), 0.05, 1).item()
        alpha = np.geomspace(0.25, 1, ti - 1) * alpha_factor

        locations = self.locations.value.swapaxes(0, 1)[:ti, :, None, ::-1]
        segments = np.concat([locations[:-1], locations[1:]], axis=2).reshape(-1, 2, 2)
        alpha = np.repeat(alpha, self.size)

        if not (isinstance(label, str) or label is None) and color is not None:
            colors = np.tile(color, ti - 1)
        else:
            colors = color
        lc = LineCollection(segments, color=colors, alpha=alpha, **kwargs)  # type: ignore
        ax.add_collection(lc)

        # trick to display label with alpha=1
        if not (isinstance(label, str) or label is None):
            for i in range(len(label)):
                if color is not None:
                    color_ = color[i]
                else:
                    color_ = color
                ax.plot(
                    self.longitudes.value[i, -1],
                    self.latitudes.value[i, -1],
                    label=label[i],
                    color=color_,
                )
        else:
            ax.plot(
                self.longitudes.value[0, -1],
                self.latitudes.value[0, -1],
                label=label,
                color=color,
            )

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
        rmse = self.map(lambda trajectory: other.rmse(trajectory))  # type: ignore
        return TimeseriesEnsemble.from_array(rmse.value, self.times.value, unit=rmse.unit, name="RMSE")

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
        separation_distance = self.map(
            lambda trajectory: other.separation_distance(trajectory)  # type: ignore
        )
        return TimeseriesEnsemble.from_array(
            separation_distance.value,
            self.times.value,
            unit=separation_distance.unit,
            name="Separation distance",
        )

    def steps(self) -> TimeseriesEnsemble:
        """
        Returns the steps of the trajectories.

        Returns
        -------
        TimeseriesEnsemble
            The steps of the trajectories.
        """
        steps = self.map(lambda trajectory: trajectory.steps())  # type: ignore
        return TimeseriesEnsemble.from_array(steps.value, self.times.value, unit=steps.unit, name="steps")

    def to_xarray(self) -> xr.Dataset:
        """
        Converts the [`pastax.trajectory.TrajectoryEnsemble`][] to a `xarray.Dataset`.

        Returns
        -------
        xr.Dataset
            The corresponding `xarray.Dataset`.
        """
        return xr.Dataset(self.to_dataarray())

    @classmethod
    def from_array(
        cls,
        values: Float[Array, "member time 2"],
        times: Float[Array, "time"],
        unit: Unit | dict[Unit, int | float] = UNIT["°"],
        id: Int[Array, ""] | None = None,
        **_: dict,
    ) -> TrajectoryEnsemble:
        """
        Creates a [`pastax.trajectory.TrajectoryEnsemble`][] from arrays of values and time points.

        Parameters
        ----------
        values : Float[Array, "member time 2"]
            The array of (latitudes, longitudes) values for the members of the trajectory ensemble.
        times : Float[Array, "time"]
            The time points for the trajectories.
        unit : Unit | dict[Unit, int | float], optional
            Unit of the trajectories locations, defaults to UNIT["°"].
        id : Int[Array, ""] | None, optional
            The ID of the trajectories, defaults to None.

        Returns
        -------
        TrajectoryEnsemble
            The corresponding [`pastax.trajectory.TrajectoryEnsemble`][].
        """
        return super().from_array(values, times, unit=unit, id=id)  # type: ignore

    def to_dataarray(self) -> dict[str, xr.DataArray]:
        """
        Converts the [`pastax.trajectory.TrajectoryEnsemble`][] to a dictionary of `xarray.DataArray`.

        Returns
        -------
        dict[str, xr.DataArray]
            A dictionary where keys are the variable names and values are the corresponding `xarray.DataArray`.
        """
        member = np.arange(self.size)
        times = self.members.times.to_datetime()
        unit = units_to_str(self.unit)

        wmo_da = xr.DataArray(
            data=self.id,
            dims=["traj", "obs"],
            coords={"id": ("traj", member)},
            name="WMO",
        )
        latitude_da = xr.DataArray(
            data=self.latitudes,
            dims=["traj", "obs"],
            coords={"id": ("traj", member), "time": ("obs", times)},
            name="lat",
            attrs={"units": unit},
        )
        longitude_da = xr.DataArray(
            data=self.longitudes,
            dims=["traj", "obs"],
            coords={"id": ("traj", member), "time": ("obs", times)},
            name="lon",
            attrs={"units": unit},
        )

        return {"WMO": wmo_da, "latitude": latitude_da, "longitude": longitude_da}
