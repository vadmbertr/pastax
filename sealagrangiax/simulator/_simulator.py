from typing import Dict

import equinox as eqx
import jax.numpy as npx
import jax.random as jrd
from jaxtyping import Array, Int

from ..input import Field
from ..trajectory import Displacement, Location, Trajectory, TrajectoryEnsemble
from ..utils import UNIT


class Simulator(eqx.Module):
    fields: Dict[str, Field]  # ssc, sst, ...
    id: str

    def __init__(self, fields: Dict[str, Field], simulator_id: str):
        self.fields: Dict[str, Field] = fields
        self.id: str = simulator_id

    @staticmethod
    @eqx.filter_jit
    def _get_minmax(
            x0: Location,
            t0: Int[Array, ""],
            ts: Int[Array, "time-1"]
    ) -> [Int[Array, ""], Int[Array, ""], Location, Location]:
        one_day = 60 * 60 * 24
        min_time = t0 - one_day
        max_time = ts[-1] + one_day  # this way we can always interpolate in time

        max_travel_distance = max_time - min_time  # m - assuming max speed is 1m/s
        max_travel_distance = Displacement(
            npx.full(2, max_travel_distance, dtype=float), unit=UNIT.meters
        ).convert_to(UNIT.degrees, x0.latitude)

        min_corner = Location(x0 - max_travel_distance)
        max_corner = Location(x0 + max_travel_distance)

        return min_time, max_time, min_corner, max_corner

    def __call__(
        self,
        x0: Location,
        t0: Int[Array, ""],
        ts: Int[Array, "time-1"],
        n_samples: int = None,
        key: jrd.PRNGKey = None
    ) -> Trajectory | TrajectoryEnsemble:
        # must return the full trajectory, including (x0, t0)
        raise NotImplementedError()
