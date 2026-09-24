"""Proper scoring rules for probabilistic (ensemble) trajectory forecasts.

Implements four scoring rules (see `Pic et al., 2025`_) for ensemble 
forecasts of shape ``(S, T, 2)`` evaluated against an observed trajectory 
``(T, 2)``:

- :func:`squared_error` — deterministic-mean squared distance.
- :func:`dawid_sebastiani` — Gaussian-likelihood-based, no kernel.
- :func:`energy_score` — kernel-based proper scoring rule (unbiased estimator).
- :func:`variogram_score` — temporal variogram score for trajectory dependence.

All scores follow the *negative orientation* convention: lower is better.

The pointwise scores (:func:`squared_error`, :func:`dawid_sebastiani`,
:func:`energy_score`) accept a ``reduce`` argument:

- ``reduce=None`` returns the per-time score of shape ``(T,)``.
- ``reduce="last"`` returns the scalar score at the final time.
- ``reduce="sum"`` returns ``(weights * score).sum()``, defaulting to a
  uniform sum when ``weights`` is ``None``. By Proposition 2 of 
  `Pic et al., 2025`_,
  a non-negative-weighted sum of proper scoring rules is itself proper.
- ``reduce="joint"`` returns the scalar joint score of the whole
  trajectory, treated as one flattened ``(T*C,)`` vector; supported by
  :func:`squared_error` and :func:`energy_score` only
  (:func:`dawid_sebastiani` raises).

The default distance kernel for :func:`squared_error`, :func:`energy_score`,
and :func:`variogram_score` is the Euclidean distance. A user may pass any
callable satisfying the broadcasting kernel contract — notably
:func:`pastax.metric.separation_distance` for great-circle distances on the
sphere. Under ``reduce="joint"`` the score always uses :func:`l2_distance` on
flattened trajectories; a custom ``kernel`` is ignored with a
:class:`UserWarning`.

:func:`variogram_score` is trajectory-level rather than pointwise in time. It
compares forecast and observed increments across selected temporal lags and
returns one scalar score.

.. warning::
    **Longitude convention / antimeridian.** The default Euclidean kernel acts
    on raw ``[lon, lat]`` coordinates and is not antimeridian-safe. For
    :func:`energy_score` and :func:`variogram_score`, pass a great-circle kernel
    such as :func:`pastax.metric.separation_distance` for geographic
    trajectories. Under ``reduce="joint"`` the score always uses :func:`l2_distance` on
    flattened trajectories; a custom ``kernel`` is ignored with a
    :class:`UserWarning`. :func:`squared_error` first computes the component-wise
    ensemble mean, so it still requires a consistent longitude convention even
    when the final distance is geodesic. :func:`dawid_sebastiani` has no kernel
    hook and likewise operates directly on raw components.

.. _`Lyons, 2020`: https://doi.org/10.2140/pjm.2020.307.383
.. _`Pic et al., 2025`: https://doi.org/10.5194/ascmo-11-23-2025
"""

import warnings
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp

from ._safe_math import safe_abs_pow, safe_sqrt
from ._types import Array, Float

__all__ = [
    "dawid_sebastiani",
    "energy_score",
    "squared_error",
    "variogram_score",
]

Reduce = Literal["last", "sum", "joint"] | None
Kernel = Callable[
    [Float[Array, "... 2"], Float[Array, "... 2"]],
    Float[Array, "..."],
]


def _validate_joint_inputs(forecast, obs) -> None:
    if forecast.ndim != 3 or obs.ndim != 2 or forecast.shape[1:] != obs.shape:
        raise ValueError(
            "reduce='joint' requires forecast of shape (S, T, C) and obs of "
            "shape (T, C) with matching (T, C); got forecast "
            f"{forecast.shape}, obs {obs.shape}."
        )


def l2_distance(
    x: Float[Array, "... 2"],
    y: Float[Array, "... 2"],
) -> Float[Array, "..."]:
    return safe_sqrt(jnp.sum((x - y) ** 2, axis=-1))


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
    raise ValueError(
        "reduce must be None, 'last', or 'sum' (or 'joint' for squared_error "
        f"and energy_score); got {reduce!r}"
    )


def dawid_sebastiani(
    forecast: Float[Array, "S T 2"],
    obs: Float[Array, "T 2"],
    *,
    reduce: Reduce = None,
    weights: Float[Array, " T"] | None = None,
) -> Float[Array, " T"] | Float[Array, ""]:
    r"""Dawid-Sebastiani score: Gaussian log-likelihood under the ensemble
    (Eq. 9 of `Pic et al., 2025`_).

    The per-time score is

    .. math::

        \mathrm{DS}_t = \log\det \Sigma_t
        + (\mu_t - y_t)^{\top}\, \Sigma_t^{-1}\, (\mu_t - y_t)

    where :math:`\Sigma_t` is the unbiased (``ddof=1``) sample covariance of
    the ensemble at time :math:`t`. Requires :math:`S \geq 3` for
    :math:`\Sigma_t` to be a.s. full-rank on :math:`\mathbb{R}^2`; for
    :math:`S \leq 2` the score is undefined (singular covariance).

    .. note::
        Not antimeridian-safe and has no kernel hook: the sample covariance and
        the ``mu - obs`` term are taken on the raw ``[lon, lat]`` components, so
        an ensemble straddling ±180° gets a spuriously inflated longitude
        variance that corrupts :math:`\Sigma_t`, its log-determinant and the
        Mahalanobis term. Keep the ensemble and observation in one consistent
        longitude convention, away from the seam — see the module-level warning.

    Args:
        forecast: Ensemble forecast, shape ``(S, T, 2)``, with ``S >= 3``.
        obs: Observed trajectory, shape ``(T, 2)``.
        reduce: See :func:`squared_error`. ``"joint"`` is NOT supported and
            raises :class:`ValueError` (the joint covariance of a flattened
            ``(T*C,)``-trajectory would require ``S >= T*C + 1`` members).
        weights: See :func:`squared_error`.

    Returns:
        Per-time score of shape ``(T,)`` or a scalar, per ``reduce``.
    """
    if reduce == "joint":
        raise ValueError(
            "dawid_sebastiani does not support reduce='joint': the joint covariance "
            "of a flattened (T*C,)-trajectory would require an ensemble of size "
            "S >= T*C + 1, which is impractically large for typical trajectories."
        )
    if forecast.shape[0] < 3:
        raise ValueError(
            "dawid_sebastiani requires an ensemble of size S >= 3 (the ddof=1 "
            f"sample covariance is singular below that); got S = {forecast.shape[0]}."
        )

    def _one_t(
        fcst_t: Float[Array, "S 2"],
        obs_t: Float[Array, "2"],
    ) -> Float[Array, ""]:
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
    r"""Energy score (Eq. 12 of `Pic et al., 2025`_)  — unbiased Monte Carlo 
    estimator.

    .. math::
        \mathrm{ES}_t = \frac{1}{S} \sum_s d\!\left(X_t^{(s)}, y_t\right)^{\alpha}
        - \frac{1}{2 S (S-1)} \sum_{s \neq s'}
        d\!\left(X_t^{(s)}, X_t^{(s')}\right)^{\alpha}

    The pairwise term is computed as a full ``(S, S)`` mean (including the
    zero diagonal) multiplied by ``S/(S-1)``, which recovers the unbiased
    off-diagonal estimator exactly.

    With ``reduce="joint"``, the score is computed on trajectories flattened
    to ``(S, T*C)`` / ``(T*C,)``: the mean ``l2_distance`` to the
    observation minus half the ``S/(S-1)``-scaled mean pairwise
    ``l2_distance``, each raised to ``alpha``. ``forecast`` and ``obs``
    must have matching ``(T, C)`` trailing shapes. A custom ``kernel`` is
    ignored (a :class:`UserWarning` is emitted) and ``weights`` is ignored.

    Let :math:`d_\alpha(x,y)=\operatorname{kernel}(x,y)^\alpha`. The energy
    score is proper when :math:`d_\alpha` is of negative type, and strictly
    proper when it is of strong negative type. For the Euclidean L2 kernel 
    this gives strict propriety for :math:`\alpha\in(0,2)`.

    .. note::
        The default :func:`l2_distance` kernel is *not* antimeridian-safe. For
        geographic trajectories, pass a great-circle kernel such as
        :func:`pastax.metric.separation_distance`.
        This does not apply under ``reduce="joint"``, where the kernel is
        ignored.

        For :math:`\alpha=1`, great-circle (angular/haversine) distance on the
        sphere is of negative type, so the corresponding energy score is
        proper. It is not of strong negative type on the full sphere, hence
        strict propriety does not hold globally. `Lyons, 2020`_ proves that a
        subset of a sphere containing at most one pair of antipodal points is
        of strong negative type; on such a support (in particular, within an
        open hemisphere) the great-circle energy score is strictly proper.

    Args:
        forecast: Ensemble forecast, shape ``(S, T, 2)``, with ``S >= 2``.
        obs: Observed trajectory, shape ``(T, 2)``.
        kernel: Broadcasting distance kernel. Defaults to :func:`l2_distance`.
        alpha: Distance exponent (typically in ``(0, 2)``). Default ``1.0``.
        reduce: See :func:`squared_error`.
        weights: See :func:`squared_error`.

    Returns:
        Per-time fair energy score of shape ``(T,)`` or a scalar, per ``reduce``.
    """
    if reduce == "joint":
        _validate_joint_inputs(forecast, obs)
        if kernel is not l2_distance:
            warnings.warn(
                "reduce='joint' scores flattened (T*C,) trajectories with the "
                "L2 distance; the custom kernel is ignored.",
                UserWarning,
                stacklevel=2,
            )
        s = forecast.shape[0]
        if s < 2:
            raise ValueError(
                "energy_score requires an ensemble of size S >= 2 (the unbiased "
                f"pairwise term divides by S - 1); got S = {s}."
            )
        flat_forecast = forecast.reshape(s, -1)
        flat_obs = obs.reshape(-1)
        observation_distances = l2_distance(flat_forecast, flat_obs) ** alpha
        bias = jnp.mean(observation_distances)
        pairwise_distances = (
            l2_distance(flat_forecast[:, None, :], flat_forecast[None, :, :]) ** alpha
        )
        dispersion = jnp.mean(pairwise_distances) * s / (s - 1)
        return bias - dispersion / 2.0
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

    With the default L2 kernel this is the squared error of the ensemble mean
    (Eq. 11 of `Pic et al., 2025`_).

    .. note::
        The default :func:`l2_distance` kernel is *not* antimeridian-safe. For
        geographic trajectories, pass a great-circle kernel such as
        :func:`pastax.metric.separation_distance`.
        This does not apply under ``reduce="joint"``, where the kernel is
        ignored.

    Args:
        forecast: Ensemble forecast, shape ``(S, T, 2)``.
        obs: Observed trajectory, shape ``(T, 2)``.
        kernel: Broadcasting distance kernel. Defaults to :func:`l2_distance`.
        reduce: Time reduction. ``None`` returns the per-time vector;
            ``"last"`` returns the scalar at the final time; ``"sum"`` returns
            the (optionally weighted) sum over time; ``"joint"`` returns the
            scalar squared L2 distance between the ensemble-mean trajectory
            and the observation, both flattened to ``(T*C,)`` (a custom
            ``kernel`` is ignored with a :class:`UserWarning`; ``weights`` is
            ignored). With the default L2 kernel, ``reduce="joint"`` equals
            the UNWEIGHTED ``reduce="sum"``.
        weights: Per-time weights for ``reduce="sum"``; ignored otherwise.

    Returns:
        Per-time score of shape ``(T,)`` or a scalar, per ``reduce``.
    """
    if reduce == "joint":
        _validate_joint_inputs(forecast, obs)
        if kernel is not l2_distance:
            warnings.warn(
                "reduce='joint' scores flattened (T*C,) trajectories with the "
                "L2 distance; the custom kernel is ignored.",
                UserWarning,
                stacklevel=2,
            )
        mu = forecast.mean(axis=0).reshape(-1)
        return l2_distance(mu, obs.reshape(-1)) ** 2
    mu = forecast.mean(axis=0)
    score_per_t = kernel(mu, obs) ** 2
    return _reduce(score_per_t, reduce, weights)


def variogram_score(
    forecast: Float[Array, "S T 2"],
    obs: Float[Array, "T 2"],
    *,
    kernel: Kernel = l2_distance,
    p: float = 0.5,
    lags: tuple[int, ...] | None = None,
    lag_weights: Float[Array, " L"] | None = None,
) -> Float[Array, ""]:
    r"""Fair temporal variogram score (Eq. 13 of `Pic et al., 2025`_) of order 
    ``p`` for ensemble trajectories.

    The population score is

    .. math::
        \mathrm{VS}_{p}
        =
        \sum_{\ell\in\mathcal L}
        \frac{a_\ell}{T-\ell}
        \sum_{i=0}^{T-\ell-1}
        \left[
        \mathbb{E}_F\!\left\{
        d(X_i,X_{i+\ell})^p
        \right\}
        -
        d(y_i,y_{i+\ell})^p
        \right]^2 .

    Here :math:`d` is ``kernel``, :math:`\mathcal L` is the selected set of
    positive temporal lags, and :math:`a_\ell` is the corresponding lag weight.

    For a finite ensemble, this function uses the *fair* (unbiased) estimator
    of each squared variogram discrepancy.

    The implementation evaluates the U-statistic in :math:`O(S)` rather than
    explicitly constructing the :math:`S\times S` off-diagonal matrix:

    .. math::
        \frac{
        \left(\sum_s R_s\right)^2 - \sum_s R_s^2
        }{S(S-1)}.

    For each lag, the score is averaged over every valid starting time before
    the lag weight is applied, so short lags do not receive more weight merely
    because they contain more valid time pairs.

    This is the variogram construction applied to the *temporal dimensions* of
    a trajectory rather than to coordinate components at a single time. It
    therefore targets temporal dependence / increment structure and complements
    the pointwise-in-time :func:`energy_score`.

    .. note::
        The default :func:`l2_distance` kernel is *not* antimeridian-safe. For
        geographic trajectories, pass a great-circle kernel such as
        :func:`pastax.metric.separation_distance`.

        The population variogram score is proper but generally not strictly
        proper: it identifies the selected pairwise transformed moments, not
        the complete joint trajectory distribution.

    Args:
        forecast: Ensemble forecast, shape ``(S, T, 2)``, with ``S >= 2``.
        obs: Observed trajectory, shape ``(T, 2)``.
        kernel: Broadcasting distance kernel between trajectory states.
            Defaults to :func:`l2_distance`.
        p: Variogram order, ``p > 0``. Defaults to ``0.5``.
        lags: Positive integer temporal lags. If ``None``, uses every lag
            ``1, ..., T - 1``. Lags are Python integers and should be static
            when the function is wrapped in :func:`jax.jit`.
        lag_weights: Non-negative weight for each entry of ``lags``. If
            ``None``, uses equal weights summing to one. Supplied weights are
            normalized to sum to one. Invalid lags raise :class:`ValueError`
            (lags must stay concrete Python integers under :func:`jax.jit` —
            close them over or use ``static_argnames``); invalid lag_weights
            raise :class:`ValueError` in eager mode.

    Returns:
        Scalar fair temporal variogram score.
    """
    s, t = forecast.shape[:2]

    if s < 2:
        raise ValueError(
            "variogram_score requires an ensemble of size S >= 2 for the fair "
            f"off-diagonal estimator; got S = {s}."
        )

    if obs.shape[0] != t:
        raise ValueError(
            "forecast and obs must have the same time dimension; "
            f"got T={t} and T_obs={obs.shape[0]}."
        )

    if lags is None:
        lags = tuple(range(1, t))

    if len(lags) == 0:
        raise ValueError("lags must contain at least one positive lag.")

    for lag in lags:
        if not isinstance(lag, int) or isinstance(lag, bool):
            raise ValueError(f"lags must be integers; got {lag!r}.")
        if not 1 <= lag <= t - 1:
            raise ValueError(
                f"lags must satisfy 1 <= lag <= T - 1 = {t - 1}; got {lag}."
            )

    if lag_weights is None:
        lag_weights = jnp.ones((len(lags),), dtype=forecast.dtype)
    else:
        lag_weights = jnp.asarray(lag_weights, dtype=forecast.dtype)
        if lag_weights.shape != (len(lags),):
            raise ValueError(
                "lag_weights must have one entry per lag; "
                f"got shape {lag_weights.shape} for {len(lags)} lags."
            )

    wmin = jnp.min(lag_weights)
    wsum = jnp.sum(lag_weights)
    if not isinstance(wmin, jax.core.Tracer):
        if float(wmin) < 0.0:
            raise ValueError("lag_weights must be non-negative.")
        if float(wsum) <= 0.0:
            raise ValueError("lag_weights must sum to a positive value.")
    lag_weights = lag_weights / wsum

    lag_scores = []
    for lag in lags:
        fcst_vgram = safe_abs_pow(
            kernel(forecast[:, :-lag], forecast[:, lag:]),
            p,
        )
        obs_vgram = safe_abs_pow(
            kernel(obs[:-lag], obs[lag:]),
            p,
        )

        residual = fcst_vgram - obs_vgram[None, :]

        residual_sum = residual.sum(axis=0)
        residual_sq_sum = jnp.sum(residual**2, axis=0)
        fair_pair_score = (
            residual_sum**2 - residual_sq_sum
        ) / (s * (s - 1))

        lag_scores.append(jnp.mean(fair_pair_score))

    return jnp.sum(lag_weights * jnp.stack(lag_scores))
