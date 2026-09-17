"""Gradient-safe versions of mathematically unstable operations."""

import jax.numpy as jnp

from ._types import Array, Float

__all__ = ["safe_abs_pow", "safe_divide", "safe_log", "safe_sqrt"]


def safe_abs_pow(x: Float[Array, "..."], p: float) -> Float[Array, "..."]:
    r"""Gradient-safe :math:`|x|^p`.

    ``|x| ** p`` has an unbounded derivative at ``x = 0`` for ``p < 1``
    (:math:`p\,|x|^{p-1} \to \infty`), which turns into ``nan`` in the
    backward pass — and a ``0 * nan`` further down (e.g. a downstream
    zero-weighted term in :func:`variogram_score`) stays ``nan``.
    The standard "double where" trick evaluates the power only where
    ``x != 0`` so the value is unchanged (``|0|^p == 0`` for ``p > 0``)
    while the gradient at exact zeros is ``0`` instead of ``nan``.
    """
    mask = x != 0.0
    return jnp.where(mask, jnp.abs(jnp.where(mask, x, 1.0)) ** p, 0.0)


def safe_divide(
    a: Float[Array, "..."], b: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Gradient-safe division.

    Returns ``a / b`` where ``b != 0`` and ``0`` elsewhere. Uses the "double
    where" trick so the backward pass never divides by zero.

    Args:
        a: Numerator array.
        b: Denominator array, broadcastable with ``a``.

    Returns:
        Elementwise ``a / b`` set to ``0`` where ``b`` is zero.
    """
    mask = b != 0.0
    return jnp.where(mask, a / jnp.where(mask, b, 1.0), 0.0)


def safe_log(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Gradient-safe natural logarithm.

    Returns ``log(x)`` where ``x > 0`` and ``-inf`` elsewhere. Uses the
    "double where" trick so the backward pass never differentiates through
    ``log`` at non-positive inputs.

    Args:
        x: Input array of any shape.

    Returns:
        Elementwise ``log(x)`` set to ``-inf`` for non-positive inputs.
    """
    mask = x > 0.0
    return jnp.where(mask, jnp.log(jnp.where(mask, x, 1.0)), -jnp.inf)


def safe_sqrt(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Gradient-safe square root.

    Returns ``sqrt(x)`` where ``x > 0`` and ``0`` elsewhere. Uses the
    standard "double where" trick so that the backward pass never differentiates
    through ``sqrt`` at ``x <= 0``, avoiding ``NaN`` gradients at the origin.

    Args:
        x: Input array of any shape.

    Returns:
        Elementwise ``sqrt(x)`` clamped to ``0`` for non-positive inputs.
    """
    mask = x > 0.0
    return jnp.where(mask, jnp.sqrt(jnp.where(mask, x, 1.0)), 0.0)
