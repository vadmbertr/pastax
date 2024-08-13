from __future__ import annotations
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

from ..utils import UNIT
from ._timeseries import Timeseries
from ._state import WHAT
from ._utils import earth_distance
from .state import Location


class Trajectory(Timeseries):
    _states: Location
    _states_type: ClassVar = Location
    id: Int[Array, ""] = None

    def __init__(
        self, locations: Float[Array, "time 2"], times: Int[Array, "time"], trajectory_id: Int[Array, ""] = None, **_
    ):
        super().__init__(locations, times, what=WHAT.location, unit=UNIT.degrees)
        self.id = trajectory_id

    @property
    @eqx.filter_jit
    def latitudes(self) -> Float[Array, "time"]:
        return self.locations[:, 0]

    @property
    @eqx.filter_jit
    def locations(self) -> Float[Array, "time 2"]:
        return self.states

    @property
    @eqx.filter_jit
    def longitudes(self) -> Float[Array, "time"]:
        return self.locations[:, 1]

    @property
    @eqx.filter_jit
    def origin(self) -> Float[Array, "2"]:
        return self.locations[0]

    @property
    @eqx.filter_jit
    def _locations(self) -> Location:
        return self._states

    @eqx.filter_jit
    def lengths(self) -> Float[Array, "time"]:
        return jnp.cumsum(self.steps())

    @eqx.filter_jit
    def liu_index(self, other: Trajectory) -> Float[Array, "time"]:
        error = self.separation_distance(other).cumsum()
        cum_lengths = self.lengths().cumsum()
        liu_index = error / cum_lengths

        return liu_index

    @eqx.filter_jit
    def mae(self, other: Trajectory) -> Float[Array, "time"]:
        error = self.separation_distance(other).cumsum()
        length = jnp.arange(self.length)  # we consider that traj starts from the same x0
        mae = error / length

        return mae

    def plot(self, ax: plt.Axes, label: str, color: str, ti: int = None) -> plt.Axes:
        if ti is None:
            ti = self.length

        alpha = jnp.geomspace(.25, 1, ti)

        locations = self.locations[:ti, None, ::-1]
        segments = jnp.concat([locations[:-1], locations[1:]], axis=1)

        lc = LineCollection(segments)  # noqa
        lc.set_color(color)
        lc.set_alpha(alpha)  # noqa
        ax.add_collection(lc)

        ax.plot(locations[-2:, 0, 0], locations[-2:, 0, 1], label=label, color=color)

        return ax

    @eqx.filter_jit
    def rmse(self, other: Trajectory) -> Float[Array, "time"]:
        error = (self.separation_distance(other) ** 2).cumsum()
        length = jnp.arange(self.length)  # we consider that traj starts from the same x0
        rmse = (error / length) ** (1 / 2)

        return rmse

    @eqx.filter_jit
    def separation_distance(self, other: Trajectory) -> Float[Array, "time"]:
        def axes_func(leaf):
            axes = None
            if eqx.is_array(leaf) and leaf.ndim > 0:
                axes = 0
            return axes

        separation_distance = eqx.filter_vmap(lambda p1, p2: p1.earth_distance(p2), in_axes=axes_func)(  # noqa
            self._locations, other._locations
        )

        return separation_distance

    @eqx.filter_jit
    def steps(self) -> Float[Array, "time"]:
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
