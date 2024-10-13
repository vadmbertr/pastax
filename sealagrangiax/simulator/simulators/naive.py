from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ...grid import Dataset
from ...trajectory import Displacement, Location
from ...utils import UNIT
from .._diffrax_simulator import DeterministicDiffrax


class Naive(DeterministicDiffrax):
    """
    Naive (consider only sea surface currents) deterministic simulator.

    Methods
    -------
    drift_term(t, y, args)
        Computes the drift term for the differential equation.
    solve(x0, t0, ts, dt0=30*60, solver=dfx.Heun())
        Solves the differential equation to simulate the trajectory.
    __call__(x0, t0, ts, dt0=30*60, solver=dfx.Heun(), n_samples=None, key=None)
        Simulates the trajectory based on the initial location, time, and time steps.
    """

    @staticmethod
    @eqx.filter_jit
    def drift_term(t: int, y: Float[Array, "2"], args: Dataset) -> Float[Array, "2"]:
        """
        Computes the drift term for the differential equation.

        Parameters
        ----------
        t : int
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        args : Dataset
            The dataset containing the physical fields (only u and v here).

        Returns
        -------
        Float[Array, "2"]
            The drift term (change in latitude and longitude).
        """
        t = jnp.asarray(t)
        dataset = args
        x = Location(y)

        u, v = dataset.interp_spatiotemporal("u", "v", time=t, latitude=x.latitude, longitude=x.longitude)  # m/s
        vu = jnp.asarray([v, u])  # scalars

        dlatlon = Displacement(vu, UNIT.meters)  # m/s
        dlatlon = dlatlon.convert_to(UNIT.degrees, x.latitude)  # °/s

        return dlatlon
