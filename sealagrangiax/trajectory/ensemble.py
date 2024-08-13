from typing import Callable, ClassVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

from ..utils import UNIT
from ._ensemble import TimeseriesEnsemble
from ._state import WHAT
from .timeseries import Timeseries, Trajectory


class TrajectoryEnsemble(TimeseriesEnsemble):
    _members: Trajectory
    _members_type: ClassVar = Trajectory

    def __init__(
            self,
            locations: Float[Array, "member time 2"],
            times: Int[Array, "time"],
            **_
    ):
        super().__init__(locations, times, what=WHAT.location, unit=UNIT.degrees)

    @property
    @eqx.filter_jit
    def id(self) -> Int[Array, "member"]:
        return self._members.id

    @property
    @eqx.filter_jit
    def latitudes(self) -> Float[Array, "member time"]:
        return self.states[:, :, 0]

    @property
    @eqx.filter_jit
    def locations(self) -> Float[Array, "member time 2"]:
        return self.states

    @property
    @eqx.filter_jit
    def longitudes(self) -> Float[Array, "member time"]:
        return self.states[:, :, 1]

    @property
    @eqx.filter_jit
    def origin(self) -> Float[Array, "2"]:
        return self.locations[0, 0]

    @eqx.filter_jit
    def crps(
            self,
            other: Trajectory,
            distance_func: Callable[[Trajectory, Trajectory], Float[Array, "time"]] = Trajectory.separation_distance
    ) -> Float[Array, "time"] | Timeseries:
        return super().crps(other, distance_func)  # noqa

    @eqx.filter_jit
    def liu_index(self, other: Trajectory) -> Float[Array, "member time"]:
        return self.map(lambda trajectory: other.liu_index(trajectory))

    @eqx.filter_jit
    def lengths(self) -> Float[Array, "member time"]:
        return self.map(lambda trajectory: trajectory.lengths())

    @eqx.filter_jit
    def mae(self, other: Trajectory) -> Float[Array, "member time"]:
        return self.map(lambda trajectory: other.mae(trajectory))

    def plot(self, ax: plt.Axes, label: str, color: str, ti: int = None) -> plt.Axes:
        if ti is None:
            ti = self.length

        alpha_factor = jnp.clip(1 / ((self.size / 10) ** 0.5), .05, 1).item()
        alpha = jnp.geomspace(.25, 1, ti) * alpha_factor

        locations = self.locations.swapaxes(0, 1)[:ti, :, None, ::-1]
        segments = jnp.concat([locations[:-1], locations[1:]], axis=2).reshape(-1, 2, 2)
        alpha = jnp.repeat(alpha, self.size)

        lc = LineCollection(segments, color=color, alpha=alpha)  # noqa
        ax.add_collection(lc)

        lc = LineCollection(segments[-self.size:, ...], color=color, alpha=alpha_factor)  # noqa
        ax.add_collection(lc)

        ax.plot(self.longitudes[0, -1], self.latitudes[0, -1], label=label, color=color)  # for label display

        return ax

    @eqx.filter_jit
    def rmse(self, other: Trajectory) -> Float[Array, "member time"]:
        return self.map(lambda trajectory: other.rmse(trajectory))

    @eqx.filter_jit
    def separation_distance(self, other: Trajectory) -> Float[Array, "member time"]:
        return self.map(lambda trajectory: other.separation_distance(trajectory))

    @eqx.filter_jit
    def steps(self) -> Float[Array, "member time"]:
        return self.map(lambda trajectory: trajectory.steps())
