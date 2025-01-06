import math
from collections.abc import Iterable

import jax.numpy as jnp
import xarray as xr
from jaxtyping import Array, Float, Scalar
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, NullFormatter

from ..trajectory import Set, Timeseries, TimeseriesEnsemble
from ..utils import seconds_to_days, UNIT, units_to_str


class Evaluation(Set):
    """
    Class for accessing and visualizing a dictionary of metric timeseries or timeseries ensemble.

    Methods
    -------
    __init__(states):
        Initializes the `Evaluation` object with a dictionary of metric timeseries or timeseries ensemble.
    get(key):
        Retrieves a metric by key.
    items():
        Returns the items of the metrics dictionary.
    keys():
        Returns the keys of the metrics dictionary.
    values():
        Returns the values of the metrics dictionary.
    plot(fig, ti):
        Plots the metrics timeseries or timeseries ensemble up to the time index `ti` on the figure `fig`.
    __getitem__(key):
        Retrieves a metric by key.
    """

    _members: dict[str, Timeseries | tuple[TimeseriesEnsemble, Timeseries, Timeseries]]

    def __init__(
        self,
        states: dict[str, Timeseries | tuple[TimeseriesEnsemble, Timeseries, Timeseries]],
    ):
        """
        Initializes the [`pastax.evaluation.Evaluation`][] object with a dictionary of metric timeseries or timeseries
        ensemble.

        Parameters
        ----------
        states : dict[str, Timeseries | tuple[TimeseriesEnsemble, Timeseries, Timeseries]]
            The initial metrics dictionary.
        """
        self._members = states
        self.size = len(states)

    def get(
        self, key: str
    ) -> Timeseries | TimeseriesEnsemble | tuple[TimeseriesEnsemble, Timeseries, Timeseries] | None:
        """
        Retrieves a metric by key.

        Parameters
        ----------
        key : str
            The key of the metric to retrieve.

        Returns
        -------
        Timeseries | tuple[TimeseriesEnsemble, Timeseries, Timeseries] | None
            The metrics corresponding to the key.
        """
        return self._members.get(key)

    def items(
        self,
    ) -> Iterable[
        tuple[
            str,
            Timeseries | TimeseriesEnsemble | tuple[TimeseriesEnsemble, Timeseries, Timeseries],
        ]
    ]:
        """
        Returns the items of the metrics dictionary.

        Returns
        -------
        tuple[str, Timeseries | tuple[TimeseriesEnsemble, Timeseries, Timeseries]]
            The items of the metrics dictionary.
        """
        return self._members.items()

    def keys(self) -> Iterable[str]:
        """
        Returns the keys of the metric timeseries dictionary.

        Returns
        -------
        tuple[str]
            The keys of the metrics dictionary.
        """
        return self._members.keys()

    def values(
        self,
    ) -> Iterable[Timeseries | TimeseriesEnsemble | tuple[TimeseriesEnsemble, Timeseries, Timeseries]]:
        """
        Returns the values of the metrics dictionary.

        Returns
        -------
        tuple[Timeseries | tuple[TimeseriesEnsemble, Timeseries, Timeseries]]
            The values of the metrics dictionary.
        """
        return self._members.values()

    def to_dataarray(self) -> dict[str, xr.DataArray]:
        """
        Converts the evaluation results to a dictionary of `xarray.DataArray`s.

        Returns
        -------
        dict[str, xr.DataArray]
            A dictionary where keys are the evaluation metric names and values are the corresponding
            `xarray.DataArray`s.
        """
        da = {}
        for key, value in self.items():
            if isinstance(value, Timeseries) or isinstance(value, TimeseriesEnsemble):
                da[key] = value.to_dataarray()
            else:
                ensemble, mean, crps = value
                da[key] = ensemble.to_dataarray().mean("member")
                da[f"{key} - mean"] = mean.to_dataarray().mean("member")
                da[f"{key} - CRPS"] = crps.to_dataarray().mean("member")

        return da

    def to_dataset(self) -> xr.Dataset:
        """
        Converts the evaluation results to a `xarray.Dataset`.

        Returns
        -------
        xr.Dataset
            A `xarray.Dataset` containing the evaluation results.
        """
        return xr.Dataset(self.to_dataarray())

    def plot(self, fig: Figure, ti: int | None = None):
        """
        Plots the metric timeseries or timeseries ensemble up to the time index `ti` on the figure `fig`.

        Parameters
        ----------
        fig : Figure
            The figure to plot on.
        ti : int | None, optional
            The time index up to which to plot. If `ti=None`, plots the full metric timeseries.
        """
        if ti is None:
            metric = next(iter(self.values()))
            if isinstance(metric, Timeseries) or isinstance(metric, TimeseriesEnsemble):
                ti = metric.length
            else:
                ti = metric[0].length

        n_metrics = self.size

        n_rows, n_cols = self.__guess_metrics_fig_layout(n_metrics)
        axs = fig.subplots(n_rows, n_cols)

        for i, metric in zip(range(n_metrics), self.values()):
            rowi = i // n_cols
            coli = i % n_cols
            add_xlabel = rowi == n_rows - 1
            add_legend = rowi == 0 and coli == n_cols - 1
            self._plot_metric(ti, metric, add_xlabel, add_legend, axs[rowi, coli])

    def _plot_metric(
        self,
        ti: int,
        metric: Timeseries | TimeseriesEnsemble | tuple[TimeseriesEnsemble, Timeseries, Timeseries],
        add_xlabel: bool,
        add_legend: bool,
        ax: Axes,
    ):
        min_values, max_values, timedelta, unit, name = self.__do_plot_metric(ti, metric, add_legend, ax)
        self.__set_axis_limits_labels(min_values, max_values, timedelta, unit, name, add_xlabel, ax)

    def __do_plot_metric(
        self,
        ti: int,
        metric: Timeseries | TimeseriesEnsemble | tuple[TimeseriesEnsemble, Timeseries, Timeseries],
        add_legend: bool,
        ax: Axes,
        has_label: bool = False,
    ) -> tuple[
        Float[Scalar, ""],
        Float[Scalar, ""],
        Float[Array, "time"],
        str | None,
        str | None,
    ]:
        if isinstance(metric, Timeseries) or isinstance(metric, TimeseriesEnsemble):
            values, timedelta, unit = self.__parse_metric(metric)

            values_ti = values[..., :ti]
            timedelta_ti = timedelta[:ti]

            min_values = jnp.nanmin(values[values != -jnp.inf])
            max_values = jnp.nanmax(values[values != jnp.inf])

            if isinstance(metric, Timeseries):
                name = None
                label = metric.name if has_label else None
                self.__plot_pair_metric(values_ti, timedelta_ti, ax, label=label)
            else:
                name = metric.name
                self.__plot_ensemble_metric(values_ti, timedelta_ti, ax)
        else:
            min_values, max_values, timedelta = (
                jnp.asarray(jnp.inf),
                -jnp.asarray(jnp.inf),
                jnp.asarray(jnp.nan),
            )
            unit, name = None, None
            for _metric in metric:
                _min_values, _max_values, timedelta, unit, _name = self.__do_plot_metric(
                    ti, _metric, add_legend, ax, has_label=True
                )

                if _min_values < min_values:
                    min_values = _min_values
                if _max_values > max_values:
                    max_values = _max_values

                if _name is not None:
                    name = _name

            if add_legend:
                ax.legend()

        return min_values, max_values, timedelta, unit, name

    @staticmethod
    def __guess_metrics_fig_layout(n_metrics: int) -> tuple[int, int]:
        min_n_cells = float("inf")
        best_layout = (0, 0)
        for n_rows in range(1, int(math.ceil(n_metrics**0.5)) + 1):
            n_cols = math.ceil(n_metrics / n_rows)
            if n_rows * n_cols >= n_metrics:
                n_cells = n_rows + n_cols
                if n_cells < min_n_cells:
                    min_n_cells = n_cells
                    best_layout = (n_rows, n_cols)
        return best_layout

    @staticmethod
    def __parse_metric(
        metric: Timeseries | TimeseriesEnsemble,
    ) -> tuple[Float[Array, "time"] | Float[Array, "member time"], Float[Array, "time"], str]:
        values: Float[Array, "time"] | Float[Array, "member time"] = metric.states.value[..., 1:, :]

        unit = {}
        for k, v in metric.unit.items():
            unit[k] = v
            if k == UNIT["m"]:
                values = UNIT["m"].convert_to(UNIT["km"], values, v)  # to km for visualization
                unit[UNIT["m"]] = v
            else:
                unit[k] = v

        times = metric.times.value
        timedelta = seconds_to_days(times - times[..., 0])
        timedelta: Float[Array, "time"] = timedelta[..., 1:]

        return values, timedelta, units_to_str(unit)

    def __plot_ensemble_metric(
        self,
        values: Float[Array, "member time"],
        timedelta: Float[Array, "time"],
        ax: Axes,
    ):
        timedelta_extended = jnp.tile(timedelta, (values.shape[0], 1))
        segments = jnp.concat([timedelta_extended[..., None], values[..., None]], axis=2)
        alpha = jnp.clip(1 / ((self.size / 10) ** (1 / 2)), 0.1, 1).item() / 2
        lc = LineCollection(segments, alpha=alpha, color="black")  # type: ignore
        ax.add_collection(lc)

    @staticmethod
    def __plot_pair_metric(
        values: Float[Array, "time"],
        timedelta: Float[Array, "time"],
        ax: Axes,
        label: str | None = None,
    ):
        kwargs = {}
        if label is not None:
            kwargs["label"] = label

        ax.plot(timedelta, values, **kwargs)

    @staticmethod
    def __set_axis_limits_labels(
        min_states: Float[Array, ""],
        max_states: Float[Array, ""],
        timedelta: Float[Array, "time"],
        unit: str | None,
        name: str | None,
        add_xlabel: bool,
        ax: Axes,
    ):
        ax.set_xlim(0, (timedelta[-1] + timedelta[0]).item())
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if add_xlabel:
            ax.set_xlabel("Elapsed days")
        else:
            ax.xaxis.set_major_formatter(NullFormatter())

        ax.set_ylim(
            (min_states - abs(min_states * 0.1)).item(),
            (max_states + abs(max_states * 0.1)).item(),
        )
        ylabel = f"{name}"
        if unit != "":
            ylabel += f" (${unit}$)"
        ax.set_ylabel(ylabel)

    def __getitem__(
        self, key: str
    ) -> Timeseries | TimeseriesEnsemble | tuple[TimeseriesEnsemble, Timeseries, Timeseries]:
        """
        Retrieves a metric by key.

        Parameters
        ----------
        key : str
            The key of the metric to retrieve.

        Returns
        -------
        Timeseries | TimeseriesEnsemble
            The metric [`pastax.trajectory.Timeseries`][] or [`pastax.trajectory.TimeseriesEnsemble`][] corresponding
            to the key.
        """
        return self._members[key]
