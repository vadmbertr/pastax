from __future__ import annotations
from typing import Callable, Tuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from .._diffrax_simulator import DeterministicDiffrax
from ...utils import meters_to_degrees
from ...grid import Dataset


def linear_ssc_vf(t: float, y: Float[Array, "2"], args: Dataset) -> Float[Array, "2"]:
    """
    Computes the drift term of the solved Ordinary Differential Equation.

    Parameters
    ----------
    t : float
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

    u, v = dataset.interp_spatiotemporal("u", "v", time=t, latitude=latitude, longitude=longitude)
    dlatlon = jnp.asarray([v, u])

    if dataset.is_spherical_mesh and dataset.is_uv_mps:
        dlatlon = meters_to_degrees(dlatlon, latitude=y[0])

    return dlatlon


class LinearSSC(DeterministicDiffrax):
    """
    Linear deterministic simulator considering only Sea Surface Currents.

    Attributes
    ----------
    id : str
        The identifier for the SmagorinskyDiffrax model (set to "linear_ssc").
        
    Methods
    -------
    ode_vf(t, y, args)
        Computes the drift term of the solved Ordinary Differential Equation.

    Notes
    -----
    In this example, the `ode_vf` attribute is only a function as the simulator does not have parameter to optimise.
    """

    ode_vf: Callable[[Float[Scalar, ""], Float[Array, "2"], Dataset], Float[Array, "2"]] = linear_ssc_vf
    id: str = eqx.field(static=True, default_factory=lambda: "linear_ssc")


class TrainableLinearSSCVF(eqx.Module):
    """
    Attributes
    ----------
    intercept : Float[Array, "2"], optional
        The intercept of the linear relation, defaults to [0, 0].
    slope : Float[Array, "2"], optional
        The slope of the linear relation, defaults to [1, 1].

    Methods
    -------
    __call__(t, y, args)
        Computes the drift term of the solved Ordinary Differential Equation.

    Notes
    -----
    As the class inherits from `eqx.Module`, its `intercept` and `slope` attributes can be treated as a trainable parameters.
    """

    intercept: Float[Array, "2"] = eqx.field(converter=lambda x: jnp.asarray(x))
    slope: Float[Array, "2"] = eqx.field(converter=lambda x: jnp.asarray(x))

    def __call__(self, t: float, y: Float[Array, "2"], args: Dataset) -> Float[Array, "2"]:
        """
        Computes the drift term of the solved Ordinary Differential Equation as the linear relation 
        `intercept` + `slope` * `[v, u]`.

        Parameters
        ----------
        t : float
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
        vu = linear_ssc_vf(t, y, args)  # °/s

        return self.intercept + self.slope * vu


class TrainableLinearSSC(DeterministicDiffrax):
    """
    Trainable (intercept and slope) linear deterministic simulator considering only Sea Surface Currents.

    Attributes
    ----------
    ode_vf : TrainableLinearSSCVF
        Computes the drift term of the solved Ordinary Differential Equation.
    id : str
        The identifier for the SmagorinskyDiffrax model (set to "trainable_linear_ssc").

    Methods
    -------
    from_param(intercept=(0, 0), slope = (1, 1))
        Creates a TrainableLinearSSC simulator with the given intercept and slope.

    Notes
    -----
    In this example, the `ode_vf` attribute is an `eqx.Module` with intercept and slope as attributes, allowing to treat them as a trainable parameters.
    """

    ode_vf: TrainableLinearSSCVF
    
    @classmethod
    def from_param(cls, intercept: Tuple[float] = (0, 0), slope: Tuple[float] = (1, 1)) -> TrainableLinearSSC:
        """
        Creates a TrainableLinearSSC simulator with the given Smagorinsky constant.

        Parameters
        ----------
        intercept : Tuple[float]
            The intercept of the linear relation.
        slope : Tuple[float]
            The slope of the linear relation.

        Returns
        -------
        TrainableLinearSSC
            The TrainableLinearSSC simulator.
        """
        return cls(ode_vf=TrainableLinearSSCVF(intercept=intercept, slope=slope), id="trainable_linear_ssc")
