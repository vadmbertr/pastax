from __future__ import annotations
from typing import Any, Callable, ClassVar

import equinox as eqx
from jaxtyping import Array, Float, Int

from ..utils import UNIT
from ._batch import Batch
from ._state import State, WHAT
from .ensemble import TrajectoryEnsemble
from .state import Time
from .timeseries import Timeseries, Trajectory


class StateBatch(Batch):
    _members: dict[str, State]
    size: int

    def __init__(
            self,
            states: dict[str, State]
    ):
        self._members = states
        self.size = len(states)

    @property
    @eqx.filter_jit
    def unit(self) -> dict[str, UNIT]:
        return dict((key, member.unit) for key, member in self._members.items())

    @property
    @eqx.filter_jit
    def value(self) -> dict[str, Float[Array, "state"]]:
        return dict((key, member.value) for key, member in self._members.items())

    @property
    @eqx.filter_jit
    def what(self) -> dict[str, WHAT]:
        return dict((key, member.what) for key, member in self._members.items())

    def __getitem__(self, index: Any) -> State:
        return self._members[index]


class TimeseriesBatch(Batch):
    _members: dict[str, Timeseries]
    _times: Time = eqx.field(static=True)
    size: int

    def __init__(
            self,
            timeseries: dict[str, Timeseries]
    ):
        self._members = timeseries
        self._times = next(iter(timeseries.values()))._times  # noqa
        self.size = len(timeseries)

    @property
    @eqx.filter_jit
    def states(self) -> dict[str, Float[Array, "time state"]]:
        return dict((key, member.states) for key, member in self._members.items())

    @property
    def times(self) -> Int[Array, "time"]:
        return self._times.value

    @property
    @eqx.filter_jit
    def unit(self) -> dict[str, UNIT]:
        return dict((key, member.unit) for key, member in self._members.items())

    @property
    @eqx.filter_jit
    def what(self) -> dict[str, WHAT]:
        return dict((key, member.what) for key, member in self._members.items())

    def __getitem__(self, index: Any) -> Timeseries:
        return self._members[index]


class TrajectoryBatch(TrajectoryEnsemble):
    _members: Trajectory
    _members_type: ClassVar = Trajectory
    size: int

    @eqx.filter_jit
    def batch_dispersion(
            self,
            distance_func: Callable[[Trajectory, Trajectory], Float[Array, "times"]] = Trajectory.separation_distance
    ) -> Float[Array, "times"]:
        return self.ensemble_dispersion(distance_func)  # noqa

    @eqx.filter_jit
    def crps(
            self,
            other: Trajectory,
            distance_func: Callable[[Trajectory, Trajectory], Float[Array, "time"]] = None
    ) -> Float[Array, "time"]:
        raise NotImplementedError

    @eqx.filter_jit
    def liu_index(self, other: Trajectory) -> Float[Array, "member time"]:
        raise NotImplementedError

    @eqx.filter_jit
    def mae(self, other: Trajectory) -> Float[Array, "member time"]:
        raise NotImplementedError

    @eqx.filter_jit
    def rmse(self, other: Trajectory) -> Float[Array, "member time"]:
        raise NotImplementedError

    @eqx.filter_jit
    def separation_distance(self, other: Trajectory) -> Float[Array, "member time"]:
        raise NotImplementedError
