r"""ODE and SDE solvers with a unified lax.scan integration loop.

Solver lineup
-------------
ODE solvers (both ``ode_step`` and ``sde_step``):

- :class:`Euler`, :class:`Heun`, :class:`RK4` — first / second / fourth-order
  explicit Runge–Kutta. ``sde_step`` interprets the term as a Stratonovich
  predictor–corrector that reuses the same Wiener increment across stages.

ODE-only solvers (raise on ``sde_step``):

- :class:`Tsit5` — Tsitouras 5(4)6 explicit RK, order 5 (fixed-step).
- :class:`Dopri5` — Dormand–Prince 5(4)7 explicit RK (FSAL), order 5
  (fixed-step).

SDE-only solvers (raise on ``ode_step``):

- :class:`EulerHeun` — diffrax-style Stratonovich predictor–corrector
  (diffusion-only predictor, Euler drift). Strong order 1.0.
- :class:`ItoMilstein`, :class:`StratonovichMilstein` — diagonal-noise Milstein
  schemes. Strong order 1.0. ``g`` must have shape ``(state_dim,)``; matrix
  diffusion raises.

State
-----
The state ``y`` may be any PyTree (a bare array is the single-leaf special
case, e.g. ``[lat, lon]``). The trajectory returned by :func:`solve` has the
same PyTree structure as ``y0`` with a leading ``n_save + 1`` axis on every
leaf. A PyTree state makes second-order dynamics natural — carry ``(x, v)`` and
let the term return ``(dx, dv) = (v, f(x, t))`` (see :func:`solve`).

Term API
--------
ODE term: ``f(t, y, args[, ctrl]) -> dy`` returns the time derivative ``dy`` as
a PyTree with the **same structure as** ``y`` (for a flat state, the velocity
``[dlat/dt, dlon/dt]`` in degrees/second).

SDE term: ``f(t, y, args[, ctrl]) -> (drift, diffusion)``. ``drift`` is a PyTree
matching ``y``; ``diffusion`` maps the Wiener increment to a ``y``-shaped tangent
and may be:

- a PyTree matching the noise structure — diagonal noise, applied leafwise as
  ``g * dW`` (for a flat state, ``g.shape == (state,)``);
- a bare 2-D array ``(state, n_noise)`` — applied as ``g @ dW``;
- a ``lineax.AbstractLinearOperator`` — applied as ``g.mv(dW)`` (general /
  matrix / cross-leaf noise; requires the optional ``lineax`` dependency).

The solver applies it as :math:`dy = \mathrm{drift}\,dt + g\,dW` with
:math:`dW = \sqrt{|dt|}\,z` drawn internally; the term never receives ``z``.
The Milstein solvers require a flat array state with diagonal ``g``.

The optional ``ctrl`` argument is present when ``controls`` is passed to
:func:`solve`; the solver slices ``controls[i]`` at each step and forwards it
to the term. The term owns all interpretation and scaling of the slice.

Noise convention
----------------
Per-step Wiener increment is :math:`dW = \sqrt{|dt|}\,z` with ``z`` a standard
normal drawn internally; the SDE term never sees ``z``. The noise is sampled to
match ``y0``'s structure by default, or an explicit ``brownian_structure``
(a PyTree of ``jax.ShapeDtypeStruct``) when the noise space differs from the
state space (e.g. driving an ``n``-dim state with an ``m``-dim Brownian motion
through a ``lineax`` operator). Passing ``key`` to :func:`solve` activates SDE
mode. A single trajectory is produced by default; pass ``n_samples > 1`` for an
ensemble of independent realisations (an extra leading ``n_samples`` axis via
internal vmap over split keys).

Backwards-in-time integration is supported for all solvers: pass a negative
``int_dt`` (and matching negative ``save_dt``) to :func:`solve`. SDE backwards
integration is not a textbook construction, but remains finite because the
solver sign-abs-normalises the :math:`\sqrt{dt}` factor.
"""

from __future__ import annotations

import abc
from typing import Callable

import equinox as eqx
import equinox.internal as eqxi  # pyright: ignore[reportMissingImports]  # checkpointed scan
import jax
import jax.numpy as jnp
import jax.random as jr

from ._types import Array, Float, Key, PyTree

__all__ = [
    "AbstractSolver",
    "Euler",
    "Heun",
    "RK4",
    "Tsit5",
    "Dopri5",
    "EulerHeun",
    "ItoMilstein",
    "StratonovichMilstein",
    "solve",
]


def _axpy(a: Float[Array, ""], x: PyTree, y: PyTree) -> PyTree:
    """Compute ``a * x + y`` leafwise for matching-structure PyTrees (``a`` scalar)."""
    return jax.tree.map(lambda xi, yi: a * xi + yi, x, y)


def _scale(a: Float[Array, ""], x: PyTree) -> PyTree:
    """Scale every leaf of ``x`` by the scalar ``a``."""
    return jax.tree.map(lambda xi: a * xi, x)


def _add(*trees: PyTree) -> PyTree:
    """Sum matching-structure PyTrees leafwise."""
    return jax.tree.map(lambda *leaves: sum(leaves), *trees)


def _rk_stage(y: PyTree, dt: Float[Array, ""], coeffs, ks) -> PyTree:
    """Runge–Kutta stage update ``y + dt * sum_i coeffs[i] * ks[i]`` (PyTree-leafwise)."""
    incr = jax.tree.map(lambda *ls: sum(c * leaf for c, leaf in zip(coeffs, ls)), *ks)
    return _axpy(dt, incr, y)


def _is_bare_array(x: PyTree) -> bool:
    """True if ``x`` is a single array leaf (the flat-state special case)."""
    return jax.tree.structure(x).num_leaves == 1 and eqx.is_array(jax.tree.leaves(x)[0])


def _apply_diffusion(g: PyTree, dW: PyTree) -> PyTree:
    """Apply a diffusion coefficient to a Wiener increment, returning a state-shaped tangent.

    Three forms are accepted:

    - A lineax ``AbstractLinearOperator`` mapping the noise space to the state
      space — applied as ``g.mv(dW)`` (general / matrix / cross-leaf noise).
    - A bare array ``g`` (flat-state fast path): 1-D ``g`` is a diagonal
      coefficient multiplied componentwise with ``dW``; 2-D ``g`` is a full
      ``(state, n_noise)`` matrix contracted with ``dW``.
    - A PyTree ``g`` matching the structure of ``dW`` — per-leaf diagonal noise,
      multiplied leafwise.
    """
    if _is_lineax_operator(g):
        return g.mv(dW)
    if _is_bare_array(g):
        return g @ dW if g.ndim == 2 else g * dW
    return jax.tree.map(jnp.multiply, g, dW)


def _is_lineax_operator(g: PyTree) -> bool:
    """True if ``g`` is a lineax ``AbstractLinearOperator`` (lazy import)."""
    try:
        import lineax  # noqa: PLC0415
    except ImportError:
        return False
    return isinstance(g, lineax.AbstractLinearOperator)


class AbstractSolver(eqx.Module):
    """Abstract base class for fixed-step ODE/SDE solvers.

    Subclasses implement :meth:`ode_step` (deterministic) and :meth:`sde_step`
    (stochastic, with a pre-sampled ``z``). Solvers that are specific to one
    mode raise :class:`NotImplementedError` from the other.
    """

    @abc.abstractmethod
    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
    ) -> Float[Array, "2"]:
        """Advance the ODE state by one step of size ``dt``.

        Args:
            term: Drift callable ``f(t, y, args) -> Float[Array, "2"]`` returning
                velocity in degrees per second.
            t: Current time, in seconds.
            y: Current state ``[lat, lon]`` in degrees.
            dt: Step size in seconds.
            args: Arbitrary fixed Pytree forwarded to ``term``.
            ctrl: Arbitrary time-varying Pytree forwarded to ``term``.

        Returns:
            Updated state ``[lat, lon]`` in degrees after one step.
        """
        ...

    @abc.abstractmethod
    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        r"""Advance the SDE state by one step using a pre-sampled ``z``.

        Args:
            term: Stochastic dynamics callable ``f(t, y, args) -> (drift, g)``.
                ``drift`` is the deterministic velocity in degrees per second;
                ``g`` is the diffusion coefficient with shape ``(2,)``
                (diagonal) or ``(2, 2)`` (full matrix). The term never sees
                ``z``; the Wiener increment is applied by the solver.
            t: Current time, in seconds.
            y: Current state ``[lat, lon]`` in degrees.
            dt: Step size in seconds.
            args: Arbitrary fixed Pytree forwarded to ``term``.
            ctrl: Arbitrary time-varying Pytree forwarded to ``term``.
            z: Standard-normal noise sample of shape ``(n_noise,)``. The Wiener
                increment used by the solver is :math:`dW = \sqrt{|dt|}\,z`.

        Returns:
            Updated state ``[lat, lon]`` in degrees after one step.
        """
        ...


class Euler(AbstractSolver):
    """Explicit Euler / Euler–Maruyama solver (first-order, fixed-step)."""

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
    ) -> Float[Array, "2"]:
        """One Euler step: ``y_new = y + term(t, y, args) * dt``."""
        return _axpy(dt, term(t, y, args, ctrl), y)

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        r"""One Euler–Maruyama step: :math:`y + \mathrm{drift}\,dt + g\,dW`."""
        f, g = term(t, y, args, ctrl)
        dW = _scale(jnp.sqrt(jnp.abs(dt)), z)
        return _add(y, _scale(dt, f), _apply_diffusion(g, dW))


class Heun(AbstractSolver):
    """Heun (explicit second-order, two-stage Runge–Kutta) solver.

    Convergence order 2 in the ODE case. The SDE step is a Stratonovich
    predictor–corrector that reuses the same ``dW`` in both stages.
    """

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
    ) -> Float[Array, "2"]:
        """One Heun (trapezoidal) step."""
        k1 = term(t, y, args, ctrl)
        k2 = term(t + dt, _axpy(dt, k1, y), args, ctrl)
        return _axpy(0.5 * dt, _add(k1, k2), y)

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        """One Stratonovich Heun step (same ``dW`` in predictor and corrector)."""
        f0, g0 = term(t, y, args, ctrl)
        dW = _scale(jnp.sqrt(jnp.abs(dt)), z)
        v0 = _add(_scale(dt, f0), _apply_diffusion(g0, dW))
        f1, g1 = term(t + dt, _add(y, v0), args, ctrl)
        v1 = _add(_scale(dt, f1), _apply_diffusion(g1, dW))
        return _axpy(0.5, _add(v0, v1), y)


class RK4(AbstractSolver):
    """Classical fourth-order Runge–Kutta solver (four stages, fixed-step).

    Convergence order 4 in the ODE case. The SDE step reuses the same ``dW``
    across all four stages, yielding a Stratonovich-consistent scheme whose
    strong order is limited by the noise structure.
    """

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
    ) -> Float[Array, "2"]:
        """One classical RK4 step."""
        half = dt * 0.5
        k1 = term(t, y, args, ctrl)
        k2 = term(t + half, _axpy(half, k1, y), args, ctrl)
        k3 = term(t + half, _axpy(half, k2, y), args, ctrl)
        k4 = term(t + dt,   _axpy(dt,   k3, y), args, ctrl)
        return _rk_stage(y, dt / 6.0, (1.0, 2.0, 2.0, 1.0), (k1, k2, k3, k4))

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        """One stochastic RK4 step (Stratonovich, single ``dW`` across stages)."""
        half = dt * 0.5
        dW = _scale(jnp.sqrt(jnp.abs(dt)), z)

        def increment(t_, y_):
            f_, g_ = term(t_, y_, args, ctrl)
            return _add(_scale(dt, f_), _apply_diffusion(g_, dW))

        v1 = increment(t,        y)
        v2 = increment(t + half, _axpy(0.5, v1, y))
        v3 = increment(t + half, _axpy(0.5, v2, y))
        v4 = increment(t + dt,   _add(y, v3))
        return _rk_stage(y, 1.0 / 6.0, (1.0, 2.0, 2.0, 1.0), (v1, v2, v3, v4))


# --- Tsit5 coefficients (Tsitouras 2011, RK5(4)6) ------------------------------
# Source: Ch. Tsitouras, "Runge–Kutta pairs of order 5(4) satisfying only the
# first column simplifying assumption", Comput. Math. Appl. 62 (2011), 770-775.
# Same values used by DifferentialEquations.jl and diffrax.
_TSIT5_A21 = 0.161
_TSIT5_A31 = -0.008480655492356989
_TSIT5_A32 = 0.335480655492357
_TSIT5_A41 = 2.8971530571054935
_TSIT5_A42 = -6.359448489975075
_TSIT5_A43 = 4.3622954328695815
_TSIT5_A51 = 5.325864828439257
_TSIT5_A52 = -11.748883564062828
_TSIT5_A53 = 7.4955393428898365
_TSIT5_A54 = -0.09249506636175525
_TSIT5_A61 = 5.86145544294642
_TSIT5_A62 = -12.92096931784711
_TSIT5_A63 = 8.159367898576159
_TSIT5_A64 = -0.071584973281401
_TSIT5_A65 = -0.028269050394068383

_TSIT5_C2 = 0.161
_TSIT5_C3 = 0.327
_TSIT5_C4 = 0.9
_TSIT5_C5 = 0.9800255409045097
_TSIT5_C6 = 1.0

_TSIT5_B1 = 0.09646076681806523
_TSIT5_B2 = 0.01
_TSIT5_B3 = 0.4798896504144996
_TSIT5_B4 = 1.379008574103742
_TSIT5_B5 = -3.290069515436081
_TSIT5_B6 = 2.324710524099774


class Tsit5(AbstractSolver):
    """Tsitouras 5(4)6 explicit Runge–Kutta (ODE-only, fixed-step, order 5).

    Six stages, no embedded error estimator (the 4th-order companion row of
    Tsitouras 2011 is unused since we are fixed-step).
    """

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
    ) -> Float[Array, "2"]:
        """One Tsit5 step (5th-order weights only)."""
        k1 = term(t, y, args, ctrl)
        k2 = term(t + _TSIT5_C2 * dt, _rk_stage(y, dt, (_TSIT5_A21,), (k1,)), args, ctrl)
        k3 = term(t + _TSIT5_C3 * dt, _rk_stage(y, dt, (_TSIT5_A31, _TSIT5_A32), (k1, k2)), args, ctrl)
        k4 = term(t + _TSIT5_C4 * dt,
                  _rk_stage(y, dt, (_TSIT5_A41, _TSIT5_A42, _TSIT5_A43), (k1, k2, k3)),
                  args, ctrl)
        k5 = term(t + _TSIT5_C5 * dt,
                  _rk_stage(y, dt, (_TSIT5_A51, _TSIT5_A52, _TSIT5_A53, _TSIT5_A54), (k1, k2, k3, k4)),
                  args, ctrl)
        k6 = term(t + _TSIT5_C6 * dt,
                  _rk_stage(y, dt, (_TSIT5_A61, _TSIT5_A62, _TSIT5_A63, _TSIT5_A64, _TSIT5_A65),
                            (k1, k2, k3, k4, k5)),
                  args, ctrl)
        return _rk_stage(
            y, dt,
            (_TSIT5_B1, _TSIT5_B2, _TSIT5_B3, _TSIT5_B4, _TSIT5_B5, _TSIT5_B6),
            (k1, k2, k3, k4, k5, k6),
        )

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        raise NotImplementedError(
            "Tsit5 is an ODE-only solver; use Euler, Heun, RK4, EulerHeun, "
            "ItoMilstein, or StratonovichMilstein for SDEs."
        )


class Dopri5(AbstractSolver):
    """Dormand–Prince 5(4)7 explicit Runge–Kutta (ODE-only, fixed-step, order 5).

    Seven stages with the first-same-as-last property; here we use the
    5th-order row only.
    """

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
    ) -> Float[Array, "2"]:
        """One Dopri5 step (5th-order weights only)."""
        k1 = term(t, y, args, ctrl)
        k2 = term(t + dt * (1.0 / 5.0), _rk_stage(y, dt, (1.0 / 5.0,), (k1,)), args, ctrl)
        k3 = term(t + dt * (3.0 / 10.0),
                  _rk_stage(y, dt, (3.0 / 40.0, 9.0 / 40.0), (k1, k2)),
                  args, ctrl)
        k4 = term(t + dt * (4.0 / 5.0),
                  _rk_stage(y, dt, (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0), (k1, k2, k3)),
                  args, ctrl)
        k5 = term(t + dt * (8.0 / 9.0),
                  _rk_stage(y, dt,
                            (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
                            (k1, k2, k3, k4)),
                  args, ctrl)
        k6 = term(t + dt,
                  _rk_stage(y, dt,
                            (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,
                             -5103.0 / 18656.0),
                            (k1, k2, k3, k4, k5)),
                  args, ctrl)
        return _rk_stage(
            y, dt,
            (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0),
            (k1, k2, k3, k4, k5, k6),
        )

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        raise NotImplementedError(
            "Dopri5 is an ODE-only solver; use Euler, Heun, RK4, EulerHeun, "
            "ItoMilstein, or StratonovichMilstein for SDEs."
        )


class EulerHeun(AbstractSolver):
    """Stochastic Euler–Heun solver (SDE-only, Stratonovich, strong order 1.0).

    Matches diffrax's ``EulerHeun`` algorithm: the predictor uses *diffusion
    only* and the drift is applied once Euler-style. Accepts both diagonal
    (``g.shape == (2,)``) and full (``g.shape == (2, 2)``) diffusion shapes.
    """

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
    ) -> Float[Array, "2"]:
        raise NotImplementedError(
            "EulerHeun is an SDE-only solver; use Euler, Heun, RK4, Tsit5, "
            "or Dopri5 for ODEs."
        )

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        """One stochastic Euler–Heun step."""
        f0, g0 = term(t, y, args, ctrl)
        dW = _scale(jnp.sqrt(jnp.abs(dt)), z)
        diff0 = _apply_diffusion(g0, dW)
        y_pred = _add(y, diff0)
        _, g1 = term(t + dt, y_pred, args, ctrl)
        diff1 = _apply_diffusion(g1, dW)
        return _add(y, _scale(dt, f0), _scale(0.5, _add(diff0, diff1)))


def _milstein_correction(
    term: Callable,
    t: Float[Array, ""],
    y: Float[Array, "2"],
    args: PyTree,
    ctrl: PyTree,
    g: Float[Array, "2"],
    dW: Float[Array, "n_noise"],
) -> Float[Array, "2"]:
    r"""Diagonal-noise Milstein cross-term :math:`\tfrac12\,g\,(\partial g_i/\partial y_i)\,dW^2`.

    Returns the :math:`\tfrac12\,g\,(\partial g_i/\partial y_i)\,dW^2` vector
    (Stratonovich form). Itô subtracts
    :math:`\tfrac12\,g\,(\partial g_i/\partial y_i)\,dt` on top.
    """
    def g_fn(y_):
        _, g_out = term(t, y_, args, ctrl)
        return g_out

    dgdy = jax.jacfwd(g_fn)(y)            # (2, 2)
    dgdy_diag = jnp.diag(dgdy)
    return 0.5 * g * dgdy_diag * dW ** 2


class ItoMilstein(AbstractSolver):
    """Itô Milstein solver (SDE-only, diagonal noise, strong order 1.0).

    Requires ``g.shape == (2,)``. Raises :class:`NotImplementedError` for
    matrix-valued ``g`` — use :class:`EulerHeun` for general noise.
    """

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
    ) -> Float[Array, "2"]:
        raise NotImplementedError(
            "ItoMilstein is an SDE-only solver; use Euler, Heun, RK4, Tsit5, "
            "or Dopri5 for ODEs."
        )

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        r"""One Itô Milstein step: :math:`y + f\,dt + g\,dW + \tfrac12\,g\,(\partial g/\partial y)\,(dW^2 - dt)`."""
        if not _is_bare_array(y):
            raise NotImplementedError(
                "ItoMilstein requires a flat array state; PyTree states are not "
                "supported. Use Euler, Heun, RK4, or EulerHeun."
            )
        f, g = term(t, y, args, ctrl)
        if g.ndim != 1:
            raise NotImplementedError(
                "ItoMilstein requires diagonal noise (g.shape == (2,)); "
                f"got g.shape == {g.shape}. Use EulerHeun for matrix diffusion."
            )
        dW = jnp.sqrt(jnp.abs(dt)) * z
        cross = _milstein_correction(term, t, y, args, ctrl, g, dW)
        def g_fn(y_):
            _, g_out = term(t, y_, args, ctrl)
            return g_out
        dgdy_diag = jnp.diag(jax.jacfwd(g_fn)(y))
        ito_drift = -0.5 * g * dgdy_diag * dt
        return y + f * dt + g * dW + cross + ito_drift


class StratonovichMilstein(AbstractSolver):
    r"""Stratonovich Milstein solver (SDE-only, diagonal noise, strong order 1.0).

    Requires ``g.shape == (2,)``. Differs from :class:`ItoMilstein` by the
    absence of the :math:`-\tfrac12\,g\,(\partial g/\partial y)\,dt`
    Itô-to-Stratonovich correction.
    """

    def ode_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
    ) -> Float[Array, "2"]:
        raise NotImplementedError(
            "StratonovichMilstein is an SDE-only solver; use Euler, Heun, RK4, "
            "Tsit5, or Dopri5 for ODEs."
        )

    def sde_step(
        self,
        term: Callable,
        t: Float[Array, ""],
        y: Float[Array, "2"],
        dt: Float[Array, ""],
        args: PyTree,
        ctrl: PyTree,
        z: Float[Array, "n_noise"],
    ) -> Float[Array, "2"]:
        r"""One Stratonovich Milstein step: :math:`y + f\,dt + g\,dW + \tfrac12\,g\,(\partial g/\partial y)\,dW^2`."""
        if not _is_bare_array(y):
            raise NotImplementedError(
                "StratonovichMilstein requires a flat array state; PyTree states are "
                "not supported. Use Euler, Heun, RK4, or EulerHeun."
            )
        f, g = term(t, y, args, ctrl)
        if g.ndim != 1:
            raise NotImplementedError(
                "StratonovichMilstein requires diagonal noise (g.shape == (2,)); "
                f"got g.shape == {g.shape}. Use EulerHeun for matrix diffusion."
            )
        dW = jnp.sqrt(jnp.abs(dt)) * z
        cross = _milstein_correction(term, t, y, args, ctrl, g, dW)
        return y + f * dt + g * dW + cross


def _sample_noise(
    key: Key[Array, ""],
    n_steps: int,
    proto: PyTree,
) -> PyTree:
    """Draw a standard-normal Wiener sequence shaped like ``proto``.

    ``proto`` is a PyTree of arrays or ``jax.ShapeDtypeStruct`` describing the
    Brownian space; each leaf yields an independent ``(n_steps, *leaf.shape)``
    draw with the leaf dtype. A single-array ``proto`` reproduces the flat-state
    ``(n_steps, *shape)`` draw.
    """
    leaves, treedef = jax.tree.flatten(proto)
    keys = jr.split(key, len(leaves))
    sampled = [
        jr.normal(k, shape=(n_steps, *leaf.shape), dtype=leaf.dtype)
        for k, leaf in zip(keys, leaves)
    ]
    return jax.tree.unflatten(treedef, sampled)


def _prepend(y0: PyTree, ys: PyTree) -> PyTree:
    """Prepend the initial state ``y0`` to a scanned trajectory ``ys`` (per-leaf)."""
    return jax.tree.map(lambda y0l, ysl: jnp.concatenate([y0l[None], ysl], axis=0), y0, ys)


def _subsample(traj: PyTree, step: int, axis: int = 0) -> PyTree:
    """Slice every ``step``-th saved state along ``axis`` (per-leaf)."""
    sl = (slice(None),) * axis + (slice(None, None, step),)
    return jax.tree.map(lambda a: a[sl], traj)


def _scan(body, init, xs, adjoint, checkpoints):
    """Run the integration loop with the chosen differentiation strategy.

    ``"checkpointed"`` uses Equinox's binomial-checkpointing scan (treeverse): low
    reverse-mode memory, but ``jax.jvp`` is not supported. ``"forward"`` uses a plain
    ``jax.lax.scan`` (no per-step checkpoint) — tangents propagate with O(1) extra
    state per step under ``jax.jvp`` / ``jax.jacfwd``, mirroring
    ``diffrax.ForwardMode``.
    """
    if adjoint == "checkpointed":
        return eqxi.scan(body, init, xs, kind="checkpointed", checkpoints=checkpoints)
    if adjoint == "forward":
        return jax.lax.scan(body, init, xs)
    raise ValueError(f'adjoint must be "checkpointed" or "forward", got {adjoint!r}.')


def _run_ode(
    term: Callable,
    y0: Float[Array, "2"],
    ts: Float[Array, " time"],
    solver: AbstractSolver,
    args: PyTree | None = None,
    controls: PyTree | None = None,
    adjoint: str = "checkpointed",
    checkpoints: int | str | None = None,
) -> Float[Array, "time 2"]:
    dt = ts[1] - ts[0]

    if args is None and controls is None:
        def body(y: Float[Array, "2"], t: Float[Array, ""]) -> tuple:
            term_fn = lambda t_, y_, a_, c_: term(t_, y_)
            y_new = solver.ode_step(term_fn, t, y, dt, None, None)
            return y_new, y_new
    elif args is not None and controls is None:
        def body(y: Float[Array, "2"], t: Float[Array, ""]) -> tuple:
            term_fn = lambda t_, y_, a_, c_: term(t_, y_, a_)
            y_new = solver.ode_step(term_fn, t, y, dt, args, None)
            return y_new, y_new
    elif args is None and controls is not None:
        def body(y: Float[Array, "2"], inputs: tuple) -> tuple:
            t, ctrl = inputs
            bound = lambda t_, y_, a_, c_: term(t_, y_, c_)
            y_new = solver.ode_step(bound, t, y, dt, None, ctrl)
            return y_new, y_new
    else:
        def body(y: Float[Array, "2"], inputs: tuple) -> tuple:
            t, ctrl = inputs
            bound = lambda t_, y_, a_, c_: term(t_, y_, a_, c_)
            y_new = solver.ode_step(bound, t, y, dt, args, ctrl)
            return y_new, y_new
        
    if controls is None:
        xs = ts[:-1]
    else:
        xs = (ts[:-1], controls)

    _, ys = _scan(body, y0, xs, adjoint, checkpoints)

    return _prepend(y0, ys)


def _run_sde(
    term: Callable,
    y0: Float[Array, "2"],
    ts: Float[Array, " time"],
    solver: AbstractSolver,
    z_seq: Float[Array, "steps 2"],
    args: PyTree | None = None,
    controls: PyTree | None = None,
    adjoint: str = "checkpointed",
    checkpoints: int | str | None = None,
) -> Float[Array, "time 2"]:
    dt = ts[1] - ts[0]

    if args is None and controls is None:
        def body(y: Float[Array, "2"], inputs: Float[Array, ""]) -> tuple:
            t, z = inputs
            term_fn = lambda t_, y_, a_, c_: term(t_, y_)
            y_new = solver.sde_step(term_fn, t, y, dt, None, None, z)
            return y_new, y_new
    elif args is not None and controls is None:
        def body(y: Float[Array, "2"], inputs: Float[Array, ""]) -> tuple:
            t, z = inputs
            term_fn = lambda t_, y_, a_, c_: term(t_, y_, a_)
            y_new = solver.sde_step(term_fn, t, y, dt, args, None, z)
            return y_new, y_new
    elif args is None and controls is not None:
        def body(y: Float[Array, "2"], inputs: tuple) -> tuple:
            t, z, ctrl = inputs
            bound = lambda t_, y_, a_, c_: term(t_, y_, c_)
            y_new = solver.sde_step(bound, t, y, dt, None, ctrl, z)
            return y_new, y_new
    else:
        def body(y: Float[Array, "2"], inputs: tuple) -> tuple:
            t, z, ctrl = inputs
            bound = lambda t_, y_, a_, c_: term(t_, y_, a_, c_)
            y_new = solver.sde_step(bound, t, y, dt, args, ctrl, z)
            return y_new, y_new
        
    if controls is None:
        xs = (ts[:-1], z_seq)
    else:
        xs = (ts[:-1], z_seq, controls)

    _, ys = _scan(body, y0, xs, adjoint, checkpoints)

    return _prepend(y0, ys)


def solve(
    term: Callable,
    y0: Float[Array, "2"],
    t0: Float[Array, ""],
    n_save: int,
    int_dt: float,
    save_dt: float,
    solver: AbstractSolver | None = None,
    args: PyTree | None = None,
    controls: PyTree | None = None,
    key: Key[Array, ""] | None = None,
    n_samples: int = 1,
    brownian_structure: PyTree | None = None,
    adjoint: str = "checkpointed",
    checkpoints: int | str | None = None,
) -> Array:
    r"""Integrate a trajectory for ``n_save`` output intervals starting at ``t0``.

    ODE mode (default, no ``key``): ``term(t, y[, args, ctrl])`` returns ``dy``, the
    time derivative as a PyTree matching ``y``.
    SDE mode (pass ``key``): ``term(t, y[, args, ctrl])`` returns ``(drift, diffusion)``;
    the solver draws a standard-normal ``z`` and applies
    :math:`dW = \sqrt{|\mathrm{int\_dt}|}\,z` internally. The optional ``ctrl``
    argument is present when ``controls`` is
    provided — the solver slices it at each step; the term owns its interpretation.

    The state ``y`` may be any PyTree (a bare array is the single-leaf case). For
    second-order dynamics ``dv = f(x, t) dt + noise``, ``dx = v dt``, carry
    ``y = (x, v)`` (e.g. a ``NamedTuple``) and return ``(v, f(x, t))`` from the
    term; put the noise on the velocity leaf only.

    The solver runs on a fine integration grid of ``n_fine = n_save * n_substeps``
    steps (where ``n_substeps = round(save_dt / int_dt)``), then slices every
    ``n_substeps`` steps to produce the ``n_save + 1`` saved states.

    **Ensemble**: pass ``n_samples > 1`` in SDE mode; the key is split internally.
    **Perturbed ODE**: use ODE+controls and
    ``jax.vmap(lambda c: solve(..., controls=c))(controls_batch)``.

    Args:
        term: Dynamics callable ``f(t, y[, args, ctrl])``. ODE: returns ``dy``
            (PyTree matching ``y``). SDE: returns ``(drift, diffusion)`` where
            ``diffusion`` is a PyTree matching the noise (diagonal), a 2-D array
            ``(state, n_noise)`` (matrix), or a ``lineax`` linear operator.
        y0: Initial state. Any PyTree; a bare array (e.g. ``[lat, lon]``, shape
            ``(2,)``) is the single-leaf case. Defines the output structure.
        t0: Start time in seconds. JAX scalar — can change between calls without
            recompilation. The implicit end time is ``t0 + n_save * save_dt``.
        n_save: Number of output intervals (static). Each output leaf has a leading
            ``n_save + 1`` axis including the initial state.
        int_dt: Integration step size in seconds (static). Use a negative value
            for backward-in-time integration.
        save_dt: Output interval in seconds (static). Must be an integer multiple
            of ``int_dt`` (same sign). ``n_substeps = round(save_dt / int_dt) >= 1``.
        solver: Solver instance. Defaults to Heun().
        args: Arbitrary fixed Pytree passed through to term (e.g. a Dataset).
        controls: Arbitrary per-step Pytree with leading axis ``n_fine``. Sliced at each
            integration step.
        key: PRNG key for SDE mode. When provided, draws a standard-normal noise
            sequence (shaped per ``brownian_structure``) and runs in SDE mode.
        n_samples: Number of independent SDE realisations (default 1). Ignored in
            ODE mode. When > 1, the key is split and trajectories are vmapped,
            adding a leading ``n_samples`` axis to every output leaf.
        brownian_structure: Optional prototype PyTree of ``jax.ShapeDtypeStruct``
            (or arrays) describing the Wiener process, used when the noise space
            differs from the state space (e.g. an ``m``-dim Brownian motion driving
            an ``n``-dim state via a ``lineax`` operator). Defaults to ``y0``'s
            structure and per-leaf shapes.
        adjoint: Differentiation strategy for the integration loop.
            ``"checkpointed"`` (default) uses binomial checkpointing (treeverse) for
            low reverse-mode memory, but is **reverse-mode only** — ``jax.jvp`` is not
            supported. ``"forward"`` uses a plain ``jax.lax.scan`` (no per-step
            checkpoint) ideal for forward-mode AD (``jax.jvp`` / ``jax.jacfwd``);
            mirrors ``diffrax.ForwardMode``. Reverse mode also works through this path
            at full O(n_fine) activation memory.
        checkpoints: Memory knob for ``adjoint="checkpointed"`` (ignored otherwise).
            ``None`` (default) forwards to ``equinox.internal.scan``'s built-in
            ``O(sqrt(n_fine))`` Stumm–Walther online schedule — the same default that
            ``diffrax.RecursiveCheckpointAdjoint`` uses, balancing memory against
            backward recompute. A smaller int (e.g. ``ceil(log2(n_fine))``) saves
            memory at the cost of more recompute; a larger int (or ``"all"`` for one
            per step) trades memory for less recompute.

    Returns:
        A PyTree with the structure of ``y0``; each leaf gains a leading
        ``n_save + 1`` axis (and an extra leading ``n_samples`` axis in SDE mode
        with ``n_samples > 1``). For a flat ``(2,)`` state this is shape
        ``(n_save + 1, 2)`` or ``(n_samples, n_save + 1, 2)``.
    """
    if solver is None:
        solver = Heun()

    if adjoint not in ("checkpointed", "forward"):
        raise ValueError(f'adjoint must be "checkpointed" or "forward", got {adjoint!r}.')

    n_substeps = round(save_dt / int_dt)
    if n_substeps < 1:
        raise ValueError(
            f"save_dt/int_dt must be >= 1 (got {n_substeps}). "
            "For backward integration both int_dt and save_dt must be negative."
        )
    if abs(n_substeps * int_dt - save_dt) > 1e-8 * abs(save_dt):
        raise ValueError(
            f"save_dt ({save_dt}) must be an integer multiple of int_dt ({int_dt})."
        )
    n_fine = n_save * n_substeps
    ts_fine = t0 + jnp.arange(n_fine + 1) * int_dt

    if key is not None:
        noise_proto = y0 if brownian_structure is None else brownian_structure
        if n_samples == 1:
            z = _sample_noise(key, n_fine, noise_proto)
            result = _run_sde(term, y0, ts_fine, solver, z, args, controls, adjoint, checkpoints)
            return _subsample(result, n_substeps)
        else:
            keys = jr.split(key, n_samples)
            z = jax.vmap(lambda k: _sample_noise(k, n_fine, noise_proto))(keys)
            result = jax.vmap(
                lambda z_: _run_sde(term, y0, ts_fine, solver, z_, args, controls, adjoint, checkpoints)
            )(z)
            return _subsample(result, n_substeps, axis=1)
    else:
        result = _run_ode(term, y0, ts_fine, solver, args, controls, adjoint, checkpoints)
        return _subsample(result, n_substeps)
