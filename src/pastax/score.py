"""Proper scoring rules for probabilistic (ensemble) trajectory forecasts.

Implements four scoring rules (see Pic et al., 2025) for ensemble forecasts
of shape ``(S, T, 2)`` evaluated against an observed trajectory ``(T, 2)``:

- :func:`squared_error` — deterministic-mean squared distance.
- :func:`dawid_sebastiani` — Gaussian-likelihood-based, no kernel.
- :func:`energy_score` — kernel-based proper scoring rule (unbiased estimator).
- :func:`variogram_score` — component-wise pairwise-difference score.

All scores follow the *negative orientation* convention: lower is better.

Each score accepts a ``reduce`` argument:

- ``reduce=None`` returns the per-time score of shape ``(T,)``.
- ``reduce="last"`` returns the scalar score at the final time.
- ``reduce="sum"`` returns ``(weights * score).sum()``, defaulting to a
  uniform sum when ``weights`` is ``None``. By Proposition 2 of Pic et al.,
  a non-negative-weighted sum of proper scoring rules is itself proper.

The default distance kernel for :func:`squared_error` and
:func:`energy_score` is the Euclidean distance. A user may pass any callable
satisfying the broadcasting kernel contract — 
notably :func:`pastax.metric.separation_distance`
for great-circle distances on the sphere.
"""

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp

from ._safe_math import safe_sqrt
from ._types import Array, Float

__all__ = [
    "squared_error",
    "dawid_sebastiani",
    "energy_score",
    "variogram_score",
]

Reduce = Literal["last", "sum"] | None
Kernel = Callable[
    [Float[Array, "... 2"], Float[Array, "... 2"]],
    Float[Array, "..."],
]


def l2_distance(
    x: Float[Array, "... 2"],
    y: Float[Array, "... 2"],
) -> Float[Array, "..."]:
    return safe_sqrt(jnp.sum((x - y) ** 2, axis=-1))


def _safe_abs_pow(x: Float[Array, "..."], p: float) -> Float[Array, "..."]:
    r"""Gradient-safe :math:`|x|^p`.

    ``|x| ** p`` has an unbounded derivative at ``x = 0`` for ``p < 1``
    (:math:`p\,|x|^{p-1} \to \infty`), which turns into ``nan`` in the
    backward pass — and a ``0 * nan`` further down (e.g. a zero
    ``component_weights`` entry in :func:`variogram_score`) stays ``nan``.
    The standard "double where" trick evaluates the power only where
    ``x != 0`` so the value is unchanged (``|0|^p == 0`` for ``p > 0``)
    while the gradient at exact zeros is ``0`` instead of ``nan``.
    """
    mask = x != 0.0
    return jnp.where(mask, jnp.abs(jnp.where(mask, x, 1.0)) ** p, 0.0)


def _reduce(
    score_per_t: Float[Array, " T"],
    reduce: Reduce,
    weights: Float[Array, " T"] | None,
) -> Float[Array, " T"] | Float[Array, ""]:
    if reduce is None:
        return score_per_t
    if reduce == "last":
        return score_per_t[-1]
    if reduce == "sum":
        if weights is None:
            return score_per_t.sum()
        return (weights * score_per_t).sum()
    raise ValueError(f"reduce must be None, 'last', or 'sum'; got {reduce!r}")


def squared_error(
    forecast: Float[Array, "S T 2"],
    obs: Float[Array, "T 2"],
    *,
    kernel: Kernel = l2_distance,
    reduce: Reduce = None,
    weights: Float[Array, " T"] | None = None,
) -> Float[Array, " T"] | Float[Array, ""]:
    r"""Squared distance between ensemble mean and observation.

    .. math::

        \mathrm{SE}_t = \operatorname{kernel}\!\left(
        \operatorname{mean}_s \mathrm{forecast}[s, t],\ \mathrm{obs}[t]\right)^2

    With the default L2 kernel this is the squared error of the ensemble
    mean (Pic et al. 2025, Eq. 11).

    Args:
        forecast: Ensemble forecast, shape ``(S, T, 2)``.
        obs: Observed trajectory, shape ``(T, 2)``.
        kernel: Broadcasting distance kernel. Defaults to :func:`l2_distance`.
        reduce: Time reduction. ``None`` returns the per-time vector;
            ``"last"`` returns the scalar at the final time; ``"sum"`` returns
            the (optionally weighted) sum over time.
        weights: Per-time weights for ``reduce="sum"``; ignored otherwise.

    Returns:
        Per-time score of shape ``(T,)`` or a scalar, per ``reduce``.
    """
    mu = forecast.mean(axis=0)
    score_per_t = kernel(mu, obs) ** 2
    return _reduce(score_per_t, reduce, weights)


def dawid_sebastiani(
    forecast: Float[Array, "S T 2"],
    obs: Float[Array, "T 2"],
    *,
    reduce: Reduce = None,
    weights: Float[Array, " T"] | None = None,
) -> Float[Array, " T"] | Float[Array, ""]:
    r"""Dawid-Sebastiani score: Gaussian log-likelihood of the observation under the ensemble.

    The per-time score is

    .. math::

        \mathrm{DS}_t = \log\det \Sigma_t
        + (\mu_t - y_t)^{\top}\, \Sigma_t^{-1}\, (\mu_t - y_t)

    where :math:`\Sigma_t` is the unbiased (``ddof=1``) sample covariance of the
    ensemble at time :math:`t`. Requires :math:`S \geq 3` for :math:`\Sigma_t` to
    be a.s. full-rank on :math:`\mathbb{R}^2`; for :math:`S \leq 2` the score is
    undefined (singular covariance).

    Args:
        forecast: Ensemble forecast, shape ``(S, T, 2)``, with ``S >= 3``.
        obs: Observed trajectory, shape ``(T, 2)``.
        reduce: See :func:`squared_error`.
        weights: See :func:`squared_error`.

    Returns:
        Per-time score of shape ``(T,)`` or a scalar, per ``reduce``.
    """

    if forecast.shape[0] < 3:
        raise ValueError(
            "dawid_sebastiani requires an ensemble of size S >= 3 (the ddof=1 "
            f"sample covariance is singular below that); got S = {forecast.shape[0]}."
        )

    def _one_t(fcst_t: Float[Array, "S 2"], obs_t: Float[Array, "2"]) -> Float[Array, ""]:
        s = fcst_t.shape[0]
        mu = fcst_t.mean(axis=0)
        centered = fcst_t - mu
        sigma = centered.T @ centered / (s - 1)
        _, logdet = jnp.linalg.slogdet(sigma)
        diff = mu - obs_t
        return logdet + diff @ jnp.linalg.solve(sigma, diff)

    score_per_t = jax.vmap(_one_t, in_axes=(1, 0))(forecast, obs)
    return _reduce(score_per_t, reduce, weights)


def energy_score(
    forecast: Float[Array, "S T 2"],
    obs: Float[Array, "T 2"],
    *,
    kernel: Kernel = l2_distance,
    alpha: float = 1.0,
    reduce: Reduce = None,
    weights: Float[Array, " T"] | None = None,
) -> Float[Array, " T"] | Float[Array, ""]:
    r"""Energy score (Pic et al. 2025, Eq. 12) — unbiased Monte Carlo estimator.

    .. math::

        \mathrm{ES}_t = \frac{1}{S} \sum_s d\!\left(X_t^{(s)}, y_t\right)^{\alpha}
        - \frac{1}{2 S (S-1)} \sum_{s \neq s'}
        d\!\left(X_t^{(s)}, X_t^{(s')}\right)^{\alpha}

    The pairwise term is computed as a full ``(S, S)`` mean (including the
    zero diagonal) multiplied by ``S/(S-1)``, which recovers the unbiased
    off-diagonal estimator exactly. Strictly proper for the L2 kernel and
    :math:`\alpha \in (0, 2)`; propriety with other kernels is not guaranteed.

    Args:
        forecast: Ensemble forecast, shape ``(S, T, 2)``, with ``S >= 2``.
        obs: Observed trajectory, shape ``(T, 2)``.
        kernel: Broadcasting distance kernel. Defaults to :func:`l2_distance`.
        alpha: Distance exponent (typically in ``(0, 2)``). Default ``1.0``.
        reduce: See :func:`squared_error`.
        weights: See :func:`squared_error`.

    Returns:
        Per-time score of shape ``(T,)`` or a scalar, per ``reduce``.
    """
    s = forecast.shape[0]
    if s < 2:
        raise ValueError(
            "energy_score requires an ensemble of size S >= 2 (the unbiased "
            f"pairwise term divides by S - 1); got S = {s}."
        )

    bias_per_t = jnp.mean(kernel(forecast, obs) ** alpha, axis=0)

    pairwise = kernel(forecast[:, None], forecast[None]) ** alpha
    disp_per_t = jnp.mean(pairwise, axis=(0, 1)) * s / (s - 1)

    score_per_t = bias_per_t - disp_per_t / 2.0
    return _reduce(score_per_t, reduce, weights)


def variogram_score(
    forecast: Float[Array, "S T 2"],
    obs: Float[Array, "T 2"],
    *,
    p: float = 2.0,
    component_weights: Float[Array, "2 2"] | None = None,
    reduce: Reduce = None,
    weights: Float[Array, " T"] | None = None,
) -> Float[Array, " T"] | Float[Array, ""]:
    r"""Variogram score of order ``p`` (Pic et al. 2025, Eq. 13).

    .. math::

        \mathrm{VS}_t = \sum_{i,j} w_{ij} \left(
        \mathbb{E}_F\!\left[\,|X_{t,i} - X_{t,j}|^{p}\right]
        - |y_{t,i} - y_{t,j}|^{p}\right)^2

    Sums over both component pairs :math:`(i, j)`; with the default
    ``component_weights = 1 - I``, the diagonal contribution (zero) is masked
    out and the off-diagonal pair is counted twice (symmetric formulation).

    Args:
        forecast: Ensemble forecast, shape ``(S, T, 2)``.
        obs: Observed trajectory, shape ``(T, 2)``.
        p: Variogram order. Default ``2.0``. Any ``p > 0`` is
            gradient-safe: the powers are evaluated through a "double
            where" so exactly-zero differences (the component diagonal,
            or ties in the data) contribute a zero gradient instead of
            ``nan`` when ``p < 1``.
        component_weights: ``(2, 2)`` non-negative weight matrix. Defaults to
            ``ones((2, 2)) - eye(2)``.
        reduce: See :func:`squared_error`.
        weights: See :func:`squared_error`.

    Returns:
        Per-time score of shape ``(T,)`` or a scalar, per ``reduce``.
    """
    if component_weights is None:
        component_weights = jnp.ones((2, 2)) - jnp.eye(2)

    diff_fcst = _safe_abs_pow(forecast[..., :, None] - forecast[..., None, :], p)
    ex_fcst = diff_fcst.mean(axis=0)
    diff_obs = _safe_abs_pow(obs[..., :, None] - obs[..., None, :], p)
    residual = ex_fcst - diff_obs

    score_per_t = (component_weights * residual ** 2).sum(axis=(-2, -1))
    return _reduce(score_per_t, reduce, weights)
