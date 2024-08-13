import math
import os

from cartopy import crs as ccrs
import cmocean.cm as cmo
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Array, Float
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..evaluation._evaluation import Evaluation  # noqa
from ..evaluation.evaluator import EnsembleEvaluator, PairEvaluator
from ..input._drifter import Drifter  # noqa
from ..simulator import Simulator
from ..trajectory import Location, Trajectory, TrajectoryEnsemble
from ..utils.unit import meters_to_kilometers


class Experiment:
    def __init__(
        self,
        simulator: Simulator,
        drifters: Drifter,
        experiment_id: str,
        data_path: str = "data"
    ):
        self.simulator = simulator
        self.drifters = drifters
        self.id = experiment_id
        self.data_path: str = data_path

    def plot(
        self,
        reference_trajectory: Trajectory,
        simulated_trajectory: Trajectory | TrajectoryEnsemble,
        evaluation: Evaluation
    ):
        outputs_path = self.__get_outputs_path(reference_trajectory)

        plot_path = f"{outputs_path}/plots"
        os.makedirs(plot_path, exist_ok=True)

        reference_lengths, simulated_lengths = Experiment.__get_lengths(reference_trajectory, simulated_trajectory)

        min_latitude, max_latitude, min_longitude, max_longitude = Experiment.__get_spatial_extent(
            reference_trajectory,
            simulated_trajectory
        )
        min_latitude = min_latitude.item()
        max_latitude = max_latitude.item()
        min_longitude = min_longitude.item()
        max_longitude = max_longitude.item()

        uv_da = self.simulator.fields["ssc"].get_uv().sel(
            lat_t=slice(min_latitude, max_latitude),
            lon_t=slice(min_longitude, max_longitude)
        )
        uv_min = float(uv_da.min())
        uv_max = float(uv_da.max())

        u_da = self.simulator.fields["ssc"].get_u(interpolate=True).sel(
            lat_t=slice(min_latitude, max_latitude),
            lon_t=slice(min_longitude, max_longitude)
        )
        v_da = self.simulator.fields["ssc"].get_v(interpolate=True).sel(
            lat_t=slice(min_latitude, max_latitude),
            lon_t=slice(min_longitude, max_longitude)
        )

        n_ti = reference_trajectory.length
        n_digits = int(math.log10(n_ti)) + 1
        fig_width = 6.4 * 2
        rel_width_pad = .05

        for ti in range(n_ti):
            t = np.datetime64(int(reference_trajectory.times[ti]), "s")

            fig = plt.figure(
                layout="constrained",
                figsize=(fig_width / (1 - rel_width_pad), 4.8)
            )
            fig.suptitle(t.item().strftime("%Y-%m-%d %H:%M:%S"))

            map_fig, eval_fig = fig.subfigures(1, 2, wspace=rel_width_pad)

            self.__plot_map(
                map_fig,
                reference_trajectory, simulated_trajectory,
                reference_lengths, simulated_lengths,
                uv_da, uv_min, uv_max,
                u_da, v_da,
                t, ti + 1
            )
            evaluation.plot(eval_fig, ti + 1)

            fig.savefig(f"{plot_path}/{str(ti).zfill(n_digits)}.pdf")
            plt.close(fig)

    def simulate_and_evaluate(
        self,
        trajectory_idx: int = None,
        n_samples: int = None,
        key: jrd.PRNGKey = None,
        plot: bool = False
    ):
        if trajectory_idx is None:
            self._simulate_and_evaluate_all(n_samples=n_samples, key=key, plot=plot)
        else:
            self._simulate_and_evaluate_one(trajectory_idx, n_samples=n_samples, key=key, plot=plot)

    def _simulate_and_evaluate_one(
        self,
        trajectory_idx: int,
        n_samples: int = None,
        key: jrd.PRNGKey = None,
        plot: bool = False
    ):
        reference_trajectory = self.drifters[trajectory_idx]

        x0 = Location(reference_trajectory.origin)
        t0 = reference_trajectory.times[0]
        ts = reference_trajectory.times[1:]

        simulated_trajectory = self.simulator(x0, t0, ts, n_samples=n_samples, key=key)

        if isinstance(simulated_trajectory, Trajectory):
            evaluator = PairEvaluator()
        else:
            evaluator = EnsembleEvaluator()
        evaluation = evaluator(reference_trajectory, simulated_trajectory)

        if plot:
            self.plot(reference_trajectory, simulated_trajectory, evaluation)

    def _simulate_and_evaluate_all(self, n_samples: int = None, key: jrd.PRNGKey = None, plot: bool = False):
        raise NotImplementedError

    @staticmethod
    @eqx.filter_jit
    def __get_lengths(
        reference_trajectory: Trajectory,
        simulated_trajectory: Trajectory
    ) -> (Float[Array, "time"], Float[Array, "time"] | Float[Array, "member time"]):
        return (meters_to_kilometers(reference_trajectory.lengths()),
                meters_to_kilometers(simulated_trajectory.lengths()))

    def __get_outputs_path(self, reference_trajectory: Trajectory) -> str:
        outputs_basepath = f"{self.data_path}/results/{np.datetime64('today')}/{self.id}"

        trajectory_id = int(reference_trajectory.id)
        trajectory_origin = reference_trajectory.origin

        return f"{outputs_basepath}/{trajectory_id}/{trajectory_origin[0]}_{trajectory_origin[1]}"

    @staticmethod
    @eqx.filter_jit
    def __get_spatial_extent(
        reference_trajectory: Trajectory,
        simulated_trajectory: Trajectory
    ) -> (Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]):
        def get_min_max(
            ref_ll: Float[Array, "time"], sim_ll: Float[Array, "time"]
        ) -> (Float[Array, ""], Float[Array, ""]):
            min_ll = jnp.minimum(ref_ll.min(), sim_ll.min()) - .5
            max_ll = jnp.maximum(ref_ll.max(), sim_ll.max()) + .5

            min_diff = 3  # ° -> ~33km
            lat_remain = min_diff - (max_ll - min_ll)

            min_ll, max_ll = jax.lax.cond(
                lat_remain > 0,
                lambda: (min_ll - lat_remain / 2, max_ll + lat_remain / 2),
                lambda: (min_ll, max_ll)
            )

            return min_ll, max_ll

        min_latitude, max_latitude = get_min_max(
            reference_trajectory.latitudes, simulated_trajectory.latitudes
        )
        min_longitude, max_longitude = get_min_max(
            reference_trajectory.longitudes, simulated_trajectory.longitudes
        )

        return min_latitude, max_latitude, min_longitude, max_longitude

    def __plot_map(
        self,
        map_fig: plt.Figure,
        reference_trajectory: Trajectory,
        simulated_trajectory: Trajectory | TrajectoryEnsemble,
        reference_lengths: Float[Array, "time"],
        simulated_lengths: Float[Array, "time"] | Float[Array, "member time"],
        uv_da: xr.DataArray,
        uv_min: float,
        uv_max: float,
        u_da: xr.DataArray,
        v_da: xr.DataArray,
        t: np.datetime64,
        ti: int
    ):
        t = t.astype("datetime64[ns]")

        mosaic = [
            ["ax_map", "ax_clb"],
            ["ax_ttd", "."]
        ]

        daxs = map_fig.subplot_mosaic(
            mosaic,
            width_ratios=(95, 5),
            height_ratios=(95, 5),
            per_subplot_kw={"ax_map": {"projection": ccrs.PlateCarree()}},
            gridspec_kw={"wspace": .01, "hspace": .01}
        )
        ax_map = daxs["ax_map"]
        ax_clb = daxs["ax_clb"]
        ax_ttd = daxs["ax_ttd"]

        ax_map = self.__plot_trajectories(reference_trajectory, simulated_trajectory, ti, ax_map)

        ax_map = self.__plot_ssc(
            uv_da.interp(time=t),
            uv_min,
            uv_max,
            u_da.interp(time=t),
            v_da.interp(time=t),
            ax_map,
            ax_clb
        )

        ax_map.coastlines()
        gl = ax_map.gridlines(draw_labels=True, x_inline=False, y_inline=False, color="None", crs=ccrs.PlateCarree())
        gl.bottom_labels = False
        gl.right_labels = False

        ax_ttd = self.__plot_total_travel_distance(reference_lengths, simulated_lengths, ti, ax_ttd)  # noqa

    @staticmethod
    def __plot_ssc(
        uv: xr.DataArray,
        uv_min: float,
        uv_max: float,
        u: xr.DataArray,
        v: xr.DataArray,
        ax: plt.Axes,
        cax: plt.Axes
    ) -> plt.Axes:
        im = ax.pcolormesh(
            uv.lon_t,
            uv.lat_t,
            uv.data,
            cmap=cmo.speed,  # noqa
            vmin=uv_min,
            vmax=uv_max,
            alpha=.5,
            shading="auto",
            transform=ccrs.PlateCarree()
        )

        n_width, n_height = uv.data.shape
        scale = (n_width + n_height) / 2
        units = "width" if n_width > n_height else "height"
        ax.quiver(u.lon_t, v.lat_t, u, v, scale=scale / 2, scale_units=units, units=units, width=2e-3)

        clb = plt.colorbar(im, cax=cax, label="$\\| \\mathbf{u} \\|$", pad=0)
        clb.ax.set_title("$m/s$")

        return ax

    @staticmethod
    def __plot_trajectories(
        reference_trajectory: Trajectory,
        simulated_trajectory: Trajectory | TrajectoryEnsemble,
        ti: int,
        ax: plt.Axes
    ) -> plt.Axes:
        ax = reference_trajectory.plot(ax, "reference", "blue", ti)
        ax = simulated_trajectory.plot(ax, "simulated", "red", ti)

        ax.legend()

        return ax

    @staticmethod
    def __plot_total_travel_distance(
        reference_lengths: Float[Array, "member time"],
        simulated_lengths: Float[Array, "time"] | Float[Array, "member time"],
        ti: int,
        ax: plt.Axes
    ) -> plt.Axes:
        max_length = max(reference_lengths.max().item(), simulated_lengths.max().item())

        ax.plot([0, reference_lengths[ti-1]], [.66, .66], color="blue")

        if simulated_lengths.ndim == 1:
            ax.plot([0, simulated_lengths[ti-1]], [.33, .33], color="red")
        else:
            hist, bin_edges = jnp.histogram(simulated_lengths[:, ti-1])

            freqs = hist / hist.sum()
            freqs /= freqs.max()

            bin_segments = jnp.column_stack([bin_edges[:-1], bin_edges[1:]])[..., None]
            y = jnp.full_like(bin_segments, .33)
            bin_segments = jnp.concat([bin_segments, y], axis=-1)

            lc = LineCollection(bin_segments, alpha=freqs, color="red")  # noqa
            ax.add_collection(lc)

        ax.set_xlim([0, max_length + max_length * .05])
        ax.set_xlabel("Travel distance ($km$)")
        ax.set_ylim([0, 1])
        ax.set_ylabel("")
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_yticks([])

        return ax
