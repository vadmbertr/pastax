import math
from typing import Dict, Tuple, Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullFormatter
import xarray as xr

from ..trajectory import Set, Timeseries, TimeseriesEnsemble
from ..utils import UNIT, WHAT, meters_to_kilometers, sq_meters_to_sq_kilometers, seconds_to_days


class Evaluation(Set):
    """
    Class that manages a dictionary of metric timeseries or timeseries ensemble and provides methods to access and
    visualise them.

    Attributes
    ----------
    _members : Dict[str, Timeseries | Sequence[TimeseriesEnsemble | Timeseries]]
        A dictionary holding the metrics.
    size : int
        The number of metrics in the set.

    Methods
    -------
    __init__(states: Dict[str, Timeseries | Sequence[TimeseriesEnsemble | Timeseries]]):
        Initializes the Evaluation object with a dictionary of metric timeseries.

    get(key: str) -> Timeseries | Sequence[TimeseriesEnsemble | Timeseries]:
        Retrieves a metric by key.
    
    items() -> Tuple[str, Timeseries | Sequence[TimeseriesEnsemble | Timeseries]]:
        Returns the items of the metrics dictionary.

    keys() -> Tuple[str]:
        Returns the keys of the metric timeseries dictionary.
    
    values() -> Tuple[Timeseries | Sequence[TimeseriesEnsemble | Timeseries]]:
        Returns the values of the metrics dictionary.

    plot(fig: plt.Figure, ti: int = None):
        Plots the metric timeseries or timeseries ensemble up to a time index on the provided figure.

    __getitem__(key: str) -> Timeseries | TimeseriesEnsemble:
        Retrieves a metric by key.
    """
    
    _members: Dict[str, Timeseries | Sequence[TimeseriesEnsemble | Timeseries]]
    size: int

    def __init__(self, states: Dict[str, Timeseries | Sequence[TimeseriesEnsemble | Timeseries]]):
        """
        Initializes the Evaluation object with a dictionary of metric timeseries.

        Parameters
        ----------
        states (Dict[str, Timeseries | Sequence[TimeseriesEnsemble | Timeseries]])
            The initial metrics dictionary.
        """
        self._members = states
        self.size = len(states)

    def get(self, key: str) -> Timeseries | Sequence[TimeseriesEnsemble | Timeseries]:
        """
        Retrieves a metric by key.

        Parameters
        ----------
        key (str)
            The key of the metric to retrieve.

        Returns
        -------
        Timeseries | Sequence[TimeseriesEnsemble | Timeseries]
            The metrics corresponding to the key.
        """
        return self._members.get(key)

    def items(self) -> Tuple[str, Timeseries | Sequence[TimeseriesEnsemble | Timeseries]]:
        """
        Returns the items of the metrics dictionary.

        Returns
        -------
        Tuple[str, Timeseries | Sequence[TimeseriesEnsemble | Timeseries]]
            The items of the metrics dictionary.
        """
        return self._members.items()

    def keys(self) -> Tuple[str]:
        """
        Returns the keys of the metric timeseries dictionary.

        Returns
        -------
        Tuple[str]
            The keys of the metrics dictionary.
        """
        return self._members.keys()

    def values(self) -> Tuple[Timeseries | Sequence[TimeseriesEnsemble | Timeseries]]:
        """
        Returns the values of the metrics dictionary.

        Returns
        -------
        Tuple[Timeseries | Sequence[TimeseriesEnsemble | Timeseries]]
            The values of the metrics dictionary.
        """
        return self._members.values()

    def to_dataarray(self) -> Dict[str, xr.DataArray]:
        """
        Converts the evaluation results to a dictionary of xarray DataArrays.

        Returns
        -------
        Dict[str, xr.DataArray]
            A dictionary where keys are the evaluation metric names and values are the corresponding xarray DataArrays.
        """
        return dict((key, value.to_dataarray()) for key, value in self.items())

    def to_dataset(self) -> xr.Dataset:
        """
        Converts the evaluation results to an xarray Dataset.

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the evaluation results.
        """
        return xr.Dataset(self.to_dataarray())

    def plot(self, fig: plt.Figure, ti: int = None):
        """
        Plots the metric timeseries or timeseries ensemble up to a time index on the provided figure.

        Parameters
        ----------
        fig (plt.Figure)
            The figure to plot on.
        ti (int, optional)
            The time index up to which to plot. If None, plots the full metric timeseries.
        """
        if ti is None:
            ti = next(iter(self.values())).length

        n_metrics = self.size

        n_rows, n_cols = self.__guess_metrics_fig_layout(n_metrics)
        axs = fig.subplots(n_rows, n_cols)

        for i, metric in zip(range(n_metrics), self.values()):
            rowi = i // n_cols
            coli = i % n_cols
            add_xlabel = rowi == n_rows - 1
            add_legend = (rowi == 0 and coli == n_cols - 1)
            self._plot_metric(ti, metric, add_xlabel, add_legend, axs[rowi, coli])

    def _plot_metric(
        self,
        ti: int,
        metric: Timeseries | Sequence[TimeseriesEnsemble | Timeseries],
        add_xlabel: bool,
        add_legend: bool,
        ax: plt.Axes
    ):
        min_values, max_values, timedelta, unit, what = self.__do_plot_metric(ti, metric, add_legend, ax)
        self.__set_axis_limits_labels(min_values, max_values, timedelta, unit, what, add_xlabel, ax)

    def __do_plot_metric(
        self,
        ti: int,
        metric: Timeseries | Sequence[TimeseriesEnsemble | Timeseries],
        add_legend: bool,
        ax: plt.Axes,
        has_label: bool = False
    ) -> Tuple[Float[Array, ""], Float[Array, ""], Float[Array, "time"], str, str]:
        if isinstance(metric, Timeseries) or isinstance(metric, TimeseriesEnsemble):
            values, timedelta, unit = self.__parse_metric(metric)

            values_ti = values[..., :ti]
            timedelta_ti = timedelta[:ti]

            min_values = jnp.nanmin(values[values != -jnp.inf])
            max_values = jnp.nanmax(values[values != jnp.inf])

            if isinstance(metric, Timeseries):
                what = None
                label = WHAT[metric.what] if has_label else None
                self.__plot_pair_metric(values_ti, timedelta_ti, ax, label=label)
            else:
                what = WHAT[metric.what]
                self.__plot_ensemble_metric(values_ti, timedelta_ti, ax)
        else:
            min_values, max_values, timedelta, unit, what = jnp.inf, -jnp.inf, None, None, None
            for _metric in metric:
                _min_values, _max_values, timedelta, unit, _what = self.__do_plot_metric(
                    ti, _metric, add_legend, ax, has_label=True
                )

                if _min_values < min_values:
                    min_values = _min_values
                if _max_values > max_values:
                    max_values = _max_values

                if _what is not None:
                    what = _what

            if add_legend:
                ax.legend()

        return min_values, max_values, timedelta, unit, what

    @staticmethod
    def __guess_metrics_fig_layout(n_metrics: int) -> Tuple[int, int]:
        min_n_cells = float("inf")
        best_layout = (0, 0)
        for n_rows in range(1, int(math.ceil(n_metrics ** 0.5)) + 1):
            n_cols = math.ceil(n_metrics / n_rows)
            if n_rows * n_cols >= n_metrics:
                n_cells = n_rows + n_cols
                if n_cells < min_n_cells:
                    min_n_cells = n_cells
                    best_layout = (n_rows, n_cols)
        return best_layout

    @staticmethod
    def __parse_metric(
        metric: Timeseries | TimeseriesEnsemble
    ) -> Tuple[Float[Array, "time"] | Float[Array, "member time"], Float[Array, "time"], str]:
        if isinstance(metric, Timeseries):
            values = metric.states[1:]
        else:
            values = metric.states[:, 1:]
        unit = metric.unit

        if unit == UNIT.meters:
            values = meters_to_kilometers(values)  # to km
            unit = UNIT.kilometers
        elif unit == UNIT.square_meters:
            values = sq_meters_to_sq_kilometers(values)
            unit = UNIT.square_kilometers

        times = metric.times
        timedelta = seconds_to_days(times - times[0])
        timedelta = timedelta[1:]

        return values, timedelta, UNIT[unit]

    def __plot_ensemble_metric(
        self,
        values: Float[Array, "member time"],
        timedelta: Float[Array, "time"],
        ax: plt.Axes
    ):
        timedelta_extended = jnp.tile(timedelta, (values.shape[0], 1))
        segments = jnp.concat([timedelta_extended[..., None], values[..., None]], axis=2)
        alpha = jnp.clip(1 / ((self.size / 10) ** (1 / 2)), .1, 1).item() / 2
        lc = LineCollection(segments, alpha=alpha, color="black")
        ax.add_collection(lc)

    @staticmethod
    def __plot_pair_metric(
        values: Float[Array, "time"],
        timedelta: Float[Array, "time"],
        ax: plt.Axes,
        label: str = None
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
        unit: str,
        what: str,
        add_xlabel: bool,
        ax: plt.Axes
    ):
        ax.set_xlim([0, timedelta[-1] + timedelta[0]])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if add_xlabel:
            ax.set_xlabel("Elapsed days")
        else:
            ax.xaxis.set_major_formatter(NullFormatter())

        ax.set_ylim([min_states - abs(min_states * .1), max_states + abs(max_states * .1)])
        ylabel = f"{what}"
        if unit != "":
            ylabel += f" (${unit}$)"
        ax.set_ylabel(ylabel)

    def __getitem__(self, key: str) -> Timeseries | TimeseriesEnsemble:
        """
        Retrieves a metric by key.

        Parameters
        ----------
            y (str): The key of the metric to retrieve.
            
        Returns
        -------
            Timeseries | TimeseriesEnsemble: The metric timeseries or timeseries ensemble corresponding to the key.
        """
        return self._members[key]
