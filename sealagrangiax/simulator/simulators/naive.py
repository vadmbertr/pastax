from __future__ import annotations
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from .._diffrax_simulator import DeterministicDiffrax
from ...grid import Dataset


def naive_vf(t: Float[Scalar, ""], y: Float[Array, "2"], args: Dataset) -> Float[Array, "2"]:
    """
    Computes the drift term of the solved Ordinary Differential Equation.

    Parameters
    ----------
    t : Float[Scalar, ""]
        The current time.
    y : Float[Array, "2"]
        The current state (latitude and longitude in degrees).
    args : Dataset
        The dataset containing the physical fields (only u and v here).

    Returns
    -------
    Float[Array, "2"]
        The drift term (change in latitude and longitude in degrees).
    """
    t = jnp.asarray(t)
    dataset = args
    latitude, longitude = y[0], y[1]

    u, v = dataset.interp_spatiotemporal("u", "v", time=t, latitude=latitude, longitude=longitude)  # °/s

    return jnp.asarray([v, u])


class Naive(DeterministicDiffrax):
    """
    Naive (consider only sea surface currents) deterministic simulator.

    Attributes
    ----------
    ode_vf(t, y, args)
        Computes the drift term of the solved Ordinary Differential Equation.
    id : str
        The identifier for the SmagorinskyDiffrax model (set to "naive").

    Notes
    -----
    In this example, the `ode_vf` attribute is only a function as the simulator does not have parameter to optimise.
    """

    ode_vf: Callable[[Float[Scalar, ""], Float[Array, "2"], Dataset], Float[Array, "2"]] = naive_vf
    id: str = eqx.field(static=True, default_factory=lambda: "naive")
