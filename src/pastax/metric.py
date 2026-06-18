"""Along-trajectory metrics for evaluating Lagrangian simulation quality.

Every metric here is a *broadcasting* function of two trajectories,
``f(y, y_ref)``, whose last two axes are ``(time, 2)`` and whose leading axes
broadcast under standard NumPy/JAX rules. A metric therefore works
transparently on a single trajectory ``(T, 2)``, an ensemble ``(S, T, 2)``, or
any batched/broadcast pair — no ``ensemble`` flag required.

Because they satisfy the same broadcasting contract as the kernels in
:mod:`pastax.score`, these metrics double as energy-score kernels, e.g.
``energy_score(forecast, obs, kernel=liu_index, reduce="last")``.
"""

import jax.numpy as jnp

from ._safe_math import safe_divide
from ._types import Array, Float
from .geo import haversine

__all__ = [
    "separation_distance",
    "normalized_separation_distance",
    "liu_index",
]


def _cumulative_reference_length(
    y_ref: Float[Array, "*#batch time 2"],
) -> Float[Array, "*#batch time"]:
    r"""Cumulative arc length of the reference trajectory, :math:`\mathrm{l_{o, t}}`.

    :math:`\mathrm{l_{o, t}}` is the path length from the start to time :math:`t`; element 0 is 0
    (no step precedes the first point). Broadcasts over leading axes.
    """
    steps = haversine(y_ref[..., 1:, :], y_ref[..., :-1, :])  # (..., time - 1)
    zero = jnp.zeros(steps.shape[:-1] + (1,), dtype=steps.dtype)
    return jnp.concatenate([zero, jnp.cumsum(steps, axis=-1)], axis=-1)


def separation_distance(
    y: Float[Array, "*batch time 2"],
    y_ref: Float[Array, "*#batch time 2"],
) -> Float[Array, "*batch time"]:
    """Point-wise great-circle distance between ``y`` and ``y_ref``.

    Args:
        y: Predicted trajectory/-ies, shape ``(..., T, 2)``.
        y_ref: Reference trajectory, shape ``(..., T, 2)``; broadcasts against ``y``.

    Returns:
        Distance at each time step in metres, shape ``(..., T)``.
    """
    return haversine(y, y_ref)


def normalized_separation_distance(
    y: Float[Array, "*batch time 2"],
    y_ref: Float[Array, "*#batch time 2"],
) -> Float[Array, "*batch time"]:
    r"""Instantaneous separation normalised by cumulative reference arc length.

    .. math::

        \mathrm{NSD}_t = \frac{\mathrm{sep\_dist}_t}{\mathrm{trav\_dist}_t}

    where :math:`\mathrm{sep\_dist}_t` is the separation distance at time :math:`t`
    and :math:`\mathrm{trav\_dist}_t` is the reference trajectory travel distance
    at time :math:`t`.

    Args:
        y: Predicted trajectory/-ies, shape ``(..., T, 2)``.
        y_ref: Reference trajectory, shape ``(..., T, 2)``; broadcasts against ``y``.

    Returns:
        Dimensionless normalised separation, shape ``(..., T)``.
    """
    return safe_divide(haversine(y, y_ref), _cumulative_reference_length(y_ref))


def liu_index(
    y: Float[Array, "*batch time 2"],
    y_ref: Float[Array, "*#batch time 2"],
) -> Float[Array, "*batch time"]:
    r"""Liu & Weisberg (2011) normalised cumulative Lagrangian separation.

    .. math::

        \mathrm{Liu}_t = \frac{\operatorname{cumsum}(\mathrm{sep\_dist})_t}
        {\operatorname{cumsum}(\mathrm{trav\_dist})_t}

    where :math:`\mathrm{sep\_dist}_t` is the separation distance at time :math:`t`
    and :math:`\mathrm{trav\_dist}_t` is the reference trajectory travel distance
    at time :math:`t`.
    The denominator is thus a double cumulative sum of the per-step distances.

    Reference: Liu & Weisberg (2011), J. Geophys. Res.

    Args:
        y: Predicted trajectory/-ies, shape ``(..., T, 2)``.
        y_ref: Reference trajectory, shape ``(..., T, 2)``; broadcasts against ``y``.

    Returns:
        Dimensionless Liu Index, shape ``(..., T)``.
    """
    cumulative_separation = jnp.cumsum(haversine(y, y_ref), axis=-1)
    cumulative_length = jnp.cumsum(_cumulative_reference_length(y_ref), axis=-1)
    return safe_divide(cumulative_separation, cumulative_length)
