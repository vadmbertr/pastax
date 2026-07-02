"""Tests for solver.py: ODE/SDE solvers and the unified solve() entry point."""

import jax
import jax.numpy as jnp
import pytest

from pastax.solver import (
    RK4,
    Dopri5,
    Euler,
    EulerHeun,
    Heun,
    ItoMilstein,
    StratonovichMilstein,
    Tsit5,
    solve,
)


# ODE term: constant velocity. Accepts the optional args/ctrl forwarded by the
# solver step (``*args``) so the same helper works for direct-step and solve().
def uniform_ode_term(dlat, dlon):
    def term(t, y, *args):
        return jnp.array([dlat, dlon])
    return term


# SDE term: constant drift + constant diagonal diffusion coefficient.
def uniform_sde_term(dlat, dlon, noise_scale=1e-6):
    def term(t, y, *args):
        f = jnp.array([dlat, dlon])
        g = jnp.full(2, noise_scale)
        return f, g
    return term


class TestSolverStep:
    def test_euler_ode_step_constant_field(self):
        solver = Euler()
        y0 = jnp.array([0.0, 0.0])
        y1 = solver.ode_step(uniform_ode_term(0.1, 0.2), jnp.array(0.0), y0,
                             jnp.array(1.0), None, None)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_heun_ode_step_constant_field(self):
        solver = Heun()
        y0 = jnp.array([0.0, 0.0])
        y1 = solver.ode_step(uniform_ode_term(0.1, 0.2), jnp.array(0.0), y0,
                             jnp.array(1.0), None, None)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_euler_sde_step_zero_noise(self):
        solver = Euler()
        y0 = jnp.array([0.0, 0.0])
        term = uniform_sde_term(0.1, 0.2, noise_scale=0.0)
        z = jnp.zeros(2)
        y1 = solver.sde_step(term, jnp.array(0.0), y0, jnp.array(1.0), None, None, z)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_heun_sde_step_zero_noise(self):
        solver = Heun()
        y0 = jnp.array([0.0, 0.0])
        term = uniform_sde_term(0.1, 0.2, noise_scale=0.0)
        z = jnp.zeros(2)
        y1 = solver.sde_step(term, jnp.array(0.0), y0, jnp.array(1.0), None, None, z)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_heun_sde_uses_same_z_both_stages(self):
        solver = Heun()
        y0 = jnp.array([0.0, 0.0])
        term = uniform_sde_term(0.1, 0.2, noise_scale=1.0)
        z = jnp.ones(2)
        y1 = solver.sde_step(term, jnp.array(0.0), y0, jnp.array(0.01), None, None, z)
        assert jnp.all(jnp.isfinite(y1))

    def test_rk4_ode_step_constant_field(self):
        solver = RK4()
        y0 = jnp.array([0.0, 0.0])
        y1 = solver.ode_step(uniform_ode_term(0.1, 0.2), jnp.array(0.0), y0,
                             jnp.array(1.0), None, None)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_rk4_sde_step_zero_noise(self):
        solver = RK4()
        y0 = jnp.array([0.0, 0.0])
        term = uniform_sde_term(0.1, 0.2, noise_scale=0.0)
        z = jnp.zeros(2)
        y1 = solver.sde_step(term, jnp.array(0.0), y0, jnp.array(1.0), None, None, z)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_euler_maruyama_matches_textbook_formula(self):
        solver = Euler()
        y0 = jnp.array([0.0, 0.0])
        dt = jnp.array(0.25)
        z = jnp.array([1.5, -0.5])
        term = uniform_sde_term(0.1, 0.2, noise_scale=2.0)
        y1 = solver.sde_step(term, jnp.array(0.0), y0, dt, None, None, z)
        expected = y0 + jnp.array([0.1, 0.2]) * dt + 2.0 * jnp.sqrt(dt) * z
        assert jnp.allclose(y1, expected)


class TestSolveODE:
    def test_uniform_field_euler(self):
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        n_save, int_dt = 100, 10.0
        T = n_save * int_dt
        traj = solve(uniform_ode_term(dlat, dlon), y0, jnp.array(0.0),
                     n_save, int_dt, int_dt, Euler())
        assert traj.shape == (n_save + 1, 2)
        assert float(traj[-1, 0]) == pytest.approx(float(y0[0]) + dlat * T, rel=1e-4)
        assert float(traj[-1, 1]) == pytest.approx(float(y0[1]) + dlon * T, rel=1e-4)

    def test_uniform_field_heun(self):
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        n_save, int_dt = 100, 10.0
        T = n_save * int_dt
        traj = solve(uniform_ode_term(dlat, dlon), y0, jnp.array(0.0),
                     n_save, int_dt, int_dt, Heun())
        assert float(traj[-1, 0]) == pytest.approx(float(y0[0]) + dlat * T, rel=1e-5)

    def test_first_point_equals_y0(self):
        y0 = jnp.array([5.0, 10.0])
        traj = solve(uniform_ode_term(0.0, 0.0), y0, jnp.array(0.0), 10, 10.0, 10.0)
        assert jnp.allclose(traj[0], y0)

    def test_no_motion(self):
        y0 = jnp.array([0.0, 0.0])
        traj = solve(uniform_ode_term(0.0, 0.0), y0, jnp.array(0.0), 10, 10.0, 10.0)
        assert jnp.allclose(traj, jnp.zeros_like(traj))

    def test_jit_compatible(self):
        y0 = jnp.array([0.0, 0.0])
        term = uniform_ode_term(1e-4, 2e-4)
        traj = jax.jit(lambda y, t0: solve(term, y, t0, 10, 10.0, 10.0))(y0, jnp.array(0.0))
        assert traj.shape == (11, 2)

    def test_t0_traced_no_recompile(self):
        def time_dep_term(t, y):
            return jnp.array([1e-4 * t, 0.0])
        fn = jax.jit(lambda t0: solve(time_dep_term, jnp.zeros(2), t0, 5, 10.0, 10.0))
        traj_a = fn(jnp.array(0.0))
        traj_b = fn(jnp.array(100.0))
        assert traj_a.shape == traj_b.shape == (6, 2)
        assert not jnp.allclose(traj_a, traj_b)

    def test_substep_matches_fine_grid(self):
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([0.0, 0.0])
        term = uniform_ode_term(dlat, dlon)
        traj_sub    = solve(term, y0, jnp.array(0.0), 10, 1.0, 4.0)
        traj_coarse = solve(term, y0, jnp.array(0.0), 10, 4.0, 4.0)
        assert jnp.allclose(traj_sub, traj_coarse, atol=1e-6)

    def test_reverse_mode_grad(self):
        y0 = jnp.array([10.0, 20.0])
        term = uniform_ode_term(1e-4, 0.0)

        def loss(y0_):
            return solve(term, y0_, jnp.array(0.0), 10, 10.0, 10.0)[-1, 0]

        g = jax.grad(loss)(y0)
        assert float(g[0]) == pytest.approx(1.0, abs=1e-5)
        assert float(g[1]) == pytest.approx(0.0, abs=1e-5)

    def test_forward_mode_jvp(self):
        y0 = jnp.array([10.0, 20.0])
        term = uniform_ode_term(1e-4, 2e-4)
        _, tangent = jax.jvp(
            lambda y: solve(term, y, jnp.array(0.0), 10, 10.0, 10.0, adjoint="forward"),
            (y0,), (jnp.ones(2),),
        )
        assert jnp.all(jnp.isfinite(tangent))

    def test_forward_mode_requires_forward_adjoint(self):
        y0 = jnp.array([10.0, 20.0])
        term = uniform_ode_term(1e-4, 2e-4)
        with pytest.raises(Exception):
            jax.jvp(
                lambda y: solve(term, y, jnp.array(0.0), 10, 10.0, 10.0),  # default checkpointed
                (y0,), (jnp.ones(2),),
            )

    def test_checkpointed_matches_forward_grad(self):
        y0 = jnp.array([10.0, 20.0])
        term = uniform_ode_term(1e-4, 2e-4)

        def loss(y0_, adjoint, checkpoints=None):
            return solve(
                term, y0_, jnp.array(0.0), 10, 1.0, 10.0,
                adjoint=adjoint, checkpoints=checkpoints,
            )[-1].sum()

        g_forward = jax.grad(lambda y: loss(y, "forward"))(y0)
        g_ckpt = jax.grad(lambda y: loss(y, "checkpointed"))(y0)
        g_ckpt_explicit = jax.grad(lambda y: loss(y, "checkpointed", checkpoints=2))(y0)
        assert jnp.allclose(g_ckpt, g_forward, atol=1e-6)
        assert jnp.allclose(g_ckpt_explicit, g_forward, atol=1e-6)

    def test_requires_no_key(self):
        y0 = jnp.zeros(2)
        traj = solve(uniform_ode_term(0.0, 0.0), y0, jnp.array(0.0), 4, 2.5, 2.5)
        assert traj.shape == (5, 2)

    def test_uniform_field_rk4(self):
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        n_save, int_dt = 100, 10.0
        T = n_save * int_dt
        traj = solve(uniform_ode_term(dlat, dlon), y0, jnp.array(0.0),
                     n_save, int_dt, int_dt, RK4())
        assert traj.shape == (n_save + 1, 2)
        assert float(traj[-1, 0]) == pytest.approx(float(y0[0]) + dlat * T, rel=1e-4)
        assert float(traj[-1, 1]) == pytest.approx(float(y0[1]) + dlon * T, rel=1e-4)

    def test_rk4_convergence_order_on_linear_field(self):
        from jax import config
        config.update("jax_enable_x64", True)
        try:
            alpha = 0.1

            def term(t, y):
                return jnp.array([alpha * y[0], 0.0], dtype=jnp.float64)

            y0 = jnp.array([1.0, 0.0], dtype=jnp.float64)
            T = 1.0
            exact = float(y0[0]) * float(jnp.exp(alpha * T))

            def err(solver, n):
                dt = T / n
                return abs(float(
                    solve(term, y0, jnp.array(0.0, dtype=jnp.float64), n, dt, dt, solver)[-1, 0]
                ) - exact)

            err_rk4_coarse = err(RK4(), 4)
            err_rk4_fine   = err(RK4(), 8)
            err_heun_coarse = err(Heun(), 4)

            assert err_rk4_coarse / max(err_rk4_fine, 1e-30) > 8.0
            assert err_rk4_coarse < err_heun_coarse
        finally:
            config.update("jax_enable_x64", False)

    def test_rk4_reverse_mode_grad(self):
        y0 = jnp.array([10.0, 20.0])
        term = uniform_ode_term(1e-4, 0.0)

        def loss(y0_):
            return solve(term, y0_, jnp.array(0.0), 10, 10.0, 10.0, RK4())[-1, 0]

        g = jax.grad(loss)(y0)
        assert float(g[0]) == pytest.approx(1.0, abs=1e-5)
        assert float(g[1]) == pytest.approx(0.0, abs=1e-5)

    # --- backwards-in-time ---

    def test_backwards_in_time_constant_field(self):
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        n_save, int_dt = 100, -1.0
        T = abs(n_save * int_dt)
        traj = solve(uniform_ode_term(dlat, dlon), y0, jnp.array(100.0),
                     n_save, int_dt, int_dt, Heun())
        assert traj.shape == (n_save + 1, 2)
        assert float(traj[-1, 0]) == pytest.approx(float(y0[0]) - dlat * T, rel=1e-5)
        assert float(traj[-1, 1]) == pytest.approx(float(y0[1]) - dlon * T, rel=1e-5)

    def test_forward_then_backward_returns_to_origin(self):
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        term = uniform_ode_term(dlat, dlon)
        fwd = solve(term, y0, jnp.array(0.0), 50, 2.0, 2.0, RK4())
        bwd = solve(term, fwd[-1], jnp.array(100.0), 50, -2.0, -2.0, RK4())
        assert jnp.allclose(bwd[-1], y0, atol=1e-8)

    def test_backwards_jit_and_grad(self):
        y0 = jnp.array([10.0, 20.0])
        term = uniform_ode_term(1e-4, 0.0)
        traj = jax.jit(lambda y, t0: solve(term, y, t0, 10, -5.0, -5.0))(y0, jnp.array(50.0))
        assert traj.shape == (11, 2)

        def loss(y0_):
            return solve(term, y0_, jnp.array(50.0), 10, -5.0, -5.0)[-1, 0]

        g = jax.grad(loss)(y0)
        assert float(g[0]) == pytest.approx(1.0, abs=1e-5)

    # --- controls ---

    def test_controls_shape(self):
        # Term uses controls; output shape must be (n_save+1, 2).
        n_save, int_dt, n_fine = 5, 10.0, 5
        controls = jnp.zeros((n_fine, 2))

        def term(t, y, ctrl):
            return ctrl

        traj = solve(term, jnp.zeros(2), jnp.array(0.0), n_save, int_dt, int_dt, controls=controls)
        assert traj.shape == (n_save + 1, 2)

    def test_controls_zero_matches_plain_ode(self):
        # With zero controls fed into drift, output matches plain ODE with same drift.
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        n_save, int_dt = 20, 10.0

        def term_with_ctrl(t, y, ctrl):
            return jnp.array([dlat, dlon]) + ctrl  # ctrl = 0 → same as plain ODE

        controls = jnp.zeros((n_save, 2))
        traj_ctrl = solve(term_with_ctrl, y0, jnp.array(0.0), n_save, int_dt, int_dt,
                          controls=controls)
        traj_plain = solve(uniform_ode_term(dlat, dlon), y0, jnp.array(0.0), n_save, int_dt, int_dt)
        assert jnp.allclose(traj_ctrl, traj_plain, atol=1e-6)

    def test_controls_nonlinear_perturbed_ode(self):
        # Controls used nonlinearly (perturbed ODE style); output differs from plain ODE.
        y0 = jnp.zeros(2)
        n_save, int_dt = 10, 1.0
        controls = jax.random.normal(jax.random.key(0), (n_save, 4))

        def term(t, y, ctrl):
            # Nonlinear use of ctrl — user owns the scaling.
            residual = 1e-3 * jnp.array([jnp.tanh(ctrl[0] + ctrl[1]), jnp.tanh(ctrl[2] - ctrl[3])])
            return jnp.array([1e-4, 2e-4]) + residual

        traj = solve(term, y0, jnp.array(0.0), n_save, int_dt, int_dt, controls=controls)
        traj_plain = solve(uniform_ode_term(1e-4, 2e-4), y0, jnp.array(0.0), n_save, int_dt, int_dt)
        assert traj.shape == (n_save + 1, 2)
        assert jnp.all(jnp.isfinite(traj))
        assert not jnp.allclose(traj, traj_plain)

    def test_controls_ensemble_via_vmap(self):
        # Perturbed-ODE ensemble: vmap solve over a batch of control arrays.
        y0 = jnp.zeros(2)
        n_save, int_dt, S = 5, 1.0, 7
        controls_batch = jax.random.normal(jax.random.key(1), (S, n_save, 2))

        def term(t, y, ctrl):
            return jnp.array([1e-4, 2e-4]) + 1e-4 * ctrl

        fn = jax.vmap(lambda c: solve(term, y0, jnp.array(0.0), n_save, int_dt, int_dt, controls=c))
        ensemble = fn(controls_batch)
        assert ensemble.shape == (S, n_save + 1, 2)

    def test_sde_with_controls_shape(self):
        # SDE+controls: both key and controls can be combined.
        y0 = jnp.zeros(2)
        n_save, int_dt = 5, 1.0

        def term(t, y, ctrl):
            drift = jnp.array([1e-4, 2e-4]) + 1e-5 * ctrl
            g = jnp.full(2, 1e-6)
            return drift, g

        controls = jnp.zeros((n_save, 2))
        traj = solve(term, y0, jnp.array(0.0), n_save, int_dt, int_dt,
                     controls=controls, key=jax.random.key(0))
        assert traj.shape == (n_save + 1, 2)
        assert jnp.all(jnp.isfinite(traj))

    def test_controls_substep(self):
        # Controls on fine grid; output sliced to n_save+1 points.
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([0.0, 0.0])
        n_save, int_dt, save_dt = 5, 1.0, 4.0
        n_fine = n_save * round(save_dt / int_dt)  # 20
        controls = jnp.zeros((n_fine, 2))

        def term(t, y, ctrl):
            return jnp.array([dlat, dlon]) + ctrl

        traj = solve(term, y0, jnp.array(0.0), n_save, int_dt, save_dt, controls=controls)
        assert traj.shape == (n_save + 1, 2)


class TestSolveSDE:
    def test_single_sample_shape(self):
        y0 = jnp.zeros(2)
        key = jax.random.key(0)
        traj = solve(uniform_sde_term(1e-4, 2e-4), y0, jnp.array(0.0), 10, 10.0, 10.0, key=key)
        assert traj.shape == (11, 2)

    def test_zero_noise_matches_ode(self):
        y0 = jnp.array([10.0, 20.0])
        key = jax.random.key(0)
        sde_traj = solve(uniform_sde_term(1e-4, 2e-4, noise_scale=0.0), y0,
                         jnp.array(0.0), 50, 10.0, 10.0, key=key)
        ode_traj = solve(uniform_ode_term(1e-4, 2e-4), y0,
                         jnp.array(0.0), 50, 10.0, 10.0)
        assert jnp.allclose(sde_traj, ode_traj, atol=1e-5)

    def test_n_samples_shape(self):
        y0 = jnp.zeros(2)
        ensemble = solve(uniform_sde_term(1e-4, 2e-4), y0,
                         jnp.array(0.0), 10, 10.0, 10.0, n_samples=7, key=jax.random.key(0))
        assert ensemble.shape == (7, 11, 2)
        assert jnp.all(jnp.isfinite(ensemble))

    def test_ensemble_mean_close_to_ode_with_small_noise(self):
        y0 = jnp.zeros(2)
        ensemble = solve(uniform_sde_term(1e-4, 2e-4, noise_scale=1e-8), y0,
                         jnp.array(0.0), 10, 10.0, 10.0, n_samples=200, key=jax.random.key(0))
        ode_traj = solve(uniform_ode_term(1e-4, 2e-4), y0, jnp.array(0.0), 10, 10.0, 10.0)
        assert jnp.allclose(ensemble.mean(axis=0), ode_traj, atol=1e-4)

    def test_jit_compatible(self):
        y0 = jnp.zeros(2)
        term = uniform_sde_term(1e-4, 2e-4)
        fn = jax.jit(lambda k: solve(term, y0, jnp.array(0.0), 10, 10.0, 10.0, key=k))
        traj = fn(jax.random.key(0))
        assert traj.shape == (11, 2)

    def test_matrix_diffusion(self):
        # g.shape == (2, 2): full matrix diffusion; z is always (2,).
        def term_mat(t, y):
            drift = jnp.zeros(2)
            g = 1e-6 * jnp.array([[1.0, 0.5], [0.5, 1.0]])
            return drift, g
        y0 = jnp.zeros(2)
        traj = solve(term_mat, y0, jnp.array(0.0), 10, 10.0, 10.0, key=jax.random.key(0))
        assert traj.shape == (11, 2)
        assert jnp.all(jnp.isfinite(traj))


# ---------------------------------------------------------------------------
# Tsit5 / Dopri5  (ODE-only)
# ---------------------------------------------------------------------------

class TestTsit5:
    def test_constant_field_step(self):
        solver = Tsit5()
        y0 = jnp.array([0.0, 0.0])
        y1 = solver.ode_step(uniform_ode_term(0.1, 0.2), jnp.array(0.0), y0,
                             jnp.array(1.0), None, None)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_uniform_field_solve(self):
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        n_save, int_dt = 100, 10.0
        T = n_save * int_dt
        traj = solve(uniform_ode_term(dlat, dlon), y0, jnp.array(0.0),
                     n_save, int_dt, int_dt, Tsit5())
        assert traj.shape == (n_save + 1, 2)
        assert float(traj[-1, 0]) == pytest.approx(float(y0[0]) + dlat * T, rel=1e-4)

    def test_sde_step_raises(self):
        with pytest.raises(NotImplementedError, match="ODE-only"):
            Tsit5().sde_step(
                uniform_sde_term(0.1, 0.2), jnp.array(0.0), jnp.zeros(2),
                jnp.array(1.0), None, None, jnp.zeros(2),
            )

    def test_jit_and_grad(self):
        y0 = jnp.array([10.0, 20.0])
        term = uniform_ode_term(1e-4, 0.0)

        def loss(y0_):
            return solve(term, y0_, jnp.array(0.0), 10, 10.0, 10.0, Tsit5())[-1, 0]

        g = jax.grad(jax.jit(loss))(y0)
        assert float(g[0]) == pytest.approx(1.0, abs=1e-5)


class TestDopri5:
    def test_constant_field_step(self):
        solver = Dopri5()
        y0 = jnp.array([0.0, 0.0])
        y1 = solver.ode_step(uniform_ode_term(0.1, 0.2), jnp.array(0.0), y0,
                             jnp.array(1.0), None, None)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_uniform_field_solve(self):
        dlat, dlon = 1e-4, 2e-4
        y0 = jnp.array([10.0, 20.0])
        n_save, int_dt = 100, 10.0
        T = n_save * int_dt
        traj = solve(uniform_ode_term(dlat, dlon), y0, jnp.array(0.0),
                     n_save, int_dt, int_dt, Dopri5())
        assert traj.shape == (n_save + 1, 2)
        assert float(traj[-1, 0]) == pytest.approx(float(y0[0]) + dlat * T, rel=1e-4)

    def test_sde_step_raises(self):
        with pytest.raises(NotImplementedError, match="ODE-only"):
            Dopri5().sde_step(
                uniform_sde_term(0.1, 0.2), jnp.array(0.0), jnp.zeros(2),
                jnp.array(1.0), None, None, jnp.zeros(2),
            )

    def test_jit_and_grad(self):
        y0 = jnp.array([10.0, 20.0])
        term = uniform_ode_term(1e-4, 0.0)

        def loss(y0_):
            return solve(term, y0_, jnp.array(0.0), 10, 10.0, 10.0, Dopri5())[-1, 0]

        g = jax.grad(jax.jit(loss))(y0)
        assert float(g[0]) == pytest.approx(1.0, abs=1e-5)


def test_tsit5_and_dopri5_fifth_order_convergence():
    from jax import config
    config.update("jax_enable_x64", True)
    try:
        alpha = 0.1

        def term(t, y):
            return jnp.array([alpha * y[0], 0.0], dtype=jnp.float64)

        y0 = jnp.array([1.0, 0.0], dtype=jnp.float64)
        T = 1.0
        exact = float(y0[0]) * float(jnp.exp(alpha * T))

        def err(solver, n):
            dt = T / n
            return abs(float(
                solve(term, y0, jnp.array(0.0, dtype=jnp.float64), n, dt, dt, solver)[-1, 0]
            ) - exact)

        for solver_cls in (Tsit5, Dopri5):
            err_coarse = err(solver_cls(), 4)
            err_fine   = err(solver_cls(), 8)
            assert err_coarse / max(err_fine, 1e-30) > 16.0
            err_rk4 = err(RK4(), 4)
            assert err_coarse < err_rk4
    finally:
        config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# EulerHeun  (SDE-only)
# ---------------------------------------------------------------------------

class TestEulerHeun:
    def test_ode_step_raises(self):
        with pytest.raises(NotImplementedError, match="SDE-only"):
            EulerHeun().ode_step(
                uniform_ode_term(0.1, 0.2), jnp.array(0.0), jnp.zeros(2),
                jnp.array(1.0), None, None,
            )

    def test_zero_diffusion_matches_euler_drift(self):
        solver = EulerHeun()
        y0 = jnp.array([0.0, 0.0])
        term = uniform_sde_term(0.1, 0.2, noise_scale=0.0)
        z = jnp.array([1.0, -1.0])
        y1 = solver.sde_step(term, jnp.array(0.0), y0, jnp.array(1.0), None, None, z)
        assert jnp.allclose(y1, jnp.array([0.1, 0.2]))

    def test_constant_g_matches_euler_maruyama(self):
        y0 = jnp.array([0.0, 0.0])
        dt = jnp.array(0.5)
        z = jnp.array([1.3, -0.4])
        term = uniform_sde_term(0.1, 0.2, noise_scale=2.0)
        y_eh = EulerHeun().sde_step(term, jnp.array(0.0), y0, dt, None, None, z)
        y_em = Euler().sde_step(term, jnp.array(0.0), y0, dt, None, None, z)
        assert jnp.allclose(y_eh, y_em)

    def test_matrix_diffusion(self):
        def term_mat(t, y, *args):
            drift = jnp.array([0.1, 0.2])
            g = jnp.array([[0.5, 0.1], [0.1, 0.5]])
            return drift, g
        y0 = jnp.zeros(2)
        z = jnp.array([1.0, -1.0])
        y1 = EulerHeun().sde_step(term_mat, jnp.array(0.0), y0, jnp.array(0.25), None, None, z)
        assert jnp.all(jnp.isfinite(y1))

    def test_full_solve(self):
        y0 = jnp.zeros(2)
        traj = solve(uniform_sde_term(1e-4, 2e-4), y0, jnp.array(0.0), 10, 10.0, 10.0,
                     EulerHeun(), key=jax.random.key(0))
        assert traj.shape == (11, 2)
        assert jnp.all(jnp.isfinite(traj))


# ---------------------------------------------------------------------------
# Milstein solvers  (SDE-only, diagonal noise)
# ---------------------------------------------------------------------------

def _linear_diffusion_term(sigma, drift=(0.0, 0.0)):
    def term(t, y, *args):
        f = jnp.array(drift)
        g = sigma * y
        return f, g
    return term


class TestItoMilstein:
    def test_ode_step_raises(self):
        with pytest.raises(NotImplementedError, match="SDE-only"):
            ItoMilstein().ode_step(
                uniform_ode_term(0.1, 0.2), jnp.array(0.0), jnp.zeros(2),
                jnp.array(1.0), None, None,
            )

    def test_constant_g_matches_euler_maruyama(self):
        y0 = jnp.array([0.0, 0.0])
        dt = jnp.array(0.25)
        z = jnp.array([1.2, -0.7])
        term = uniform_sde_term(0.1, 0.2, noise_scale=2.0)
        y_im = ItoMilstein().sde_step(term, jnp.array(0.0), y0, dt, None, None, z)
        y_em = Euler().sde_step(term, jnp.array(0.0), y0, dt, None, None, z)
        assert jnp.allclose(y_im, y_em)

    def test_state_dependent_g_differs_from_stratonovich_by_ito_drift(self):
        sigma = 0.3
        y0 = jnp.array([1.0, 2.0])
        dt = jnp.array(0.1)
        z = jnp.array([0.5, -0.5])
        term = _linear_diffusion_term(sigma)
        y_ito  = ItoMilstein().sde_step(term, jnp.array(0.0), y0, dt, None, None, z)
        y_str  = StratonovichMilstein().sde_step(term, jnp.array(0.0), y0, dt, None, None, z)
        expected_diff = -0.5 * sigma**2 * y0 * dt
        assert jnp.allclose(y_ito - y_str, expected_diff)

    def test_matrix_g_raises(self):
        def term_mat(t, y, *args):
            return jnp.zeros(2), jnp.eye(2)
        with pytest.raises(NotImplementedError, match="diagonal"):
            ItoMilstein().sde_step(
                term_mat, jnp.array(0.0), jnp.zeros(2), jnp.array(1.0), None, None, jnp.zeros(2),
            )

    def test_full_solve(self):
        sigma = 0.1
        y0 = jnp.array([1.0, 1.0])
        traj = solve(_linear_diffusion_term(sigma), y0, jnp.array(0.0), 10, 0.1, 0.1,
                     ItoMilstein(), key=jax.random.key(0))
        assert traj.shape == (11, 2)
        assert jnp.all(jnp.isfinite(traj))


class TestStratonovichMilstein:
    def test_ode_step_raises(self):
        with pytest.raises(NotImplementedError, match="SDE-only"):
            StratonovichMilstein().ode_step(
                uniform_ode_term(0.1, 0.2), jnp.array(0.0), jnp.zeros(2),
                jnp.array(1.0), None, None,
            )

    def test_constant_g_has_no_correction(self):
        y0 = jnp.array([0.0, 0.0])
        dt = jnp.array(0.25)
        z = jnp.array([1.2, -0.7])
        term = uniform_sde_term(0.1, 0.2, noise_scale=2.0)
        y_sm = StratonovichMilstein().sde_step(term, jnp.array(0.0), y0, dt, None, None, z)
        y_em = Euler().sde_step(term, jnp.array(0.0), y0, dt, None, None, z)
        assert jnp.allclose(y_sm, y_em)

    def test_matrix_g_raises(self):
        def term_mat(t, y, *args):
            return jnp.zeros(2), jnp.eye(2)
        with pytest.raises(NotImplementedError, match="diagonal"):
            StratonovichMilstein().sde_step(
                term_mat, jnp.array(0.0), jnp.zeros(2), jnp.array(1.0), None, None, jnp.zeros(2),
            )

    def test_full_solve(self):
        sigma = 0.1
        y0 = jnp.array([1.0, 1.0])
        traj = solve(_linear_diffusion_term(sigma), y0, jnp.array(0.0), 10, 0.1, 0.1,
                     StratonovichMilstein(), key=jax.random.key(0))
        assert traj.shape == (11, 2)
        assert jnp.all(jnp.isfinite(traj))


# ---------------------------------------------------------------------------
# PyTree state: second-order dynamics  (dv = f dt + noise, dx = v dt)
# ---------------------------------------------------------------------------

from typing import NamedTuple  # noqa: E402


class State(NamedTuple):
    x: jnp.ndarray  # position
    v: jnp.ndarray  # velocity


def _second_order_ode_term(accel):
    # dx = v, dv = accel(x, v)
    def term(t, y, *args):
        return State(x=y.v, v=accel(y.x, y.v))
    return term


def _second_order_sde_term(accel, sigma):
    # drift = (v, accel); diagonal diffusion on the velocity leaf only
    def term(t, y, *args):
        drift = State(x=y.v, v=accel(y.x, y.v))
        diff = State(x=jnp.zeros_like(y.x), v=jnp.full_like(y.v, sigma))
        return drift, diff
    return term


class TestPyTreeStateODE:
    def test_free_particle_matches_closed_form(self):
        # accel = 0, constant velocity -> x(t) = x0 + v0 t.
        x0 = jnp.array([10.0, 20.0])
        v0 = jnp.array([1e-3, -2e-3])
        y0 = State(x=x0, v=v0)
        n_save, dt = 50, 10.0
        T = n_save * dt
        traj = solve(_second_order_ode_term(lambda x, v: jnp.zeros_like(v)),
                     y0, jnp.array(0.0), n_save, dt, dt, RK4())
        assert isinstance(traj, State)
        assert traj.x.shape == (n_save + 1, 2)
        assert traj.v.shape == (n_save + 1, 2)
        assert jnp.allclose(traj.x[-1], x0 + v0 * T, atol=1e-6)
        assert jnp.allclose(traj.v, v0[None], atol=1e-8)  # velocity unchanged

    def test_harmonic_oscillator_energy_bounded(self):
        # accel = -omega^2 x ; RK4 should keep the orbit bounded over one period.
        omega = 0.2
        y0 = State(x=jnp.array([1.0, 0.0]), v=jnp.array([0.0, 0.0]))
        traj = solve(_second_order_ode_term(lambda x, v: -(omega ** 2) * x),
                     y0, jnp.array(0.0), 200, 0.05, 0.05, RK4())
        radius = jnp.sqrt(traj.x[:, 0] ** 2 + (traj.v[:, 0] / omega) ** 2)
        assert jnp.allclose(radius, radius[0], atol=1e-3)

    def test_jit_and_grad_through_pytree_state(self):
        y0 = State(x=jnp.array([0.0, 0.0]), v=jnp.array([1.0, 0.0]))

        def loss(v0x):
            y0_ = State(x=y0.x, v=y0.v.at[0].set(v0x))
            traj = solve(_second_order_ode_term(lambda x, v: jnp.zeros_like(v)),
                         y0_, jnp.array(0.0), 10, 1.0, 1.0, RK4())
            return traj.x[-1, 0]

        g = jax.grad(jax.jit(loss))(2.0)
        assert float(g) == pytest.approx(10.0, abs=1e-5)  # d x_T / d v0 = T


class TestPyTreeStateSDE:
    def test_zero_noise_matches_ode(self):
        def accel(x, v):
            return -0.01 * v
        y0 = State(x=jnp.array([0.0, 0.0]), v=jnp.array([0.1, -0.2]))
        # With sigma=0, EulerHeun reduces to a first-order Euler drift step, so
        # compare against the matching Euler ODE integrator.
        sde = solve(_second_order_sde_term(accel, sigma=0.0), y0, jnp.array(0.0),
                    20, 1.0, 1.0, EulerHeun(), key=jax.random.key(0))
        ode = solve(_second_order_ode_term(accel), y0, jnp.array(0.0),
                    20, 1.0, 1.0, Euler())
        assert jnp.allclose(sde.x, ode.x, atol=1e-5)
        assert jnp.allclose(sde.v, ode.v, atol=1e-5)

    def test_noise_on_velocity_only_position_is_integral(self):
        # With accel = 0 and diffusion only on v, position is the exact running
        # integral of velocity: x_{k+1} = x_k + v_k * dt (Euler).
        sigma = 0.3
        dt = 1.0
        y0 = State(x=jnp.zeros(2), v=jnp.zeros(2))
        traj = solve(_second_order_sde_term(lambda x, v: jnp.zeros_like(v), sigma),
                     y0, jnp.array(0.0), 30, dt, dt, Euler(), key=jax.random.key(1))
        x, v = traj.x, traj.v
        # reconstruct position from the (saved) velocity via forward Euler
        recon = jnp.cumsum(jnp.concatenate([jnp.zeros((1, 2)), v[:-1] * dt], axis=0), axis=0)
        assert jnp.allclose(x, recon, atol=1e-5)

    def test_integrated_brownian_variance_grows(self):
        # Position variance of integrated Brownian velocity grows super-linearly.
        sigma = 0.5
        y0 = State(x=jnp.zeros(2), v=jnp.zeros(2))
        ens = solve(_second_order_sde_term(lambda x, v: jnp.zeros_like(v), sigma),
                    y0, jnp.array(0.0), 40, 1.0, 1.0, Euler(),
                    key=jax.random.key(2), n_samples=400)
        var_x = ens.x[:, :, 0].var(axis=0)
        assert var_x[-1] > var_x[len(var_x) // 2] > var_x[1]

    def test_n_samples_shape(self):
        y0 = State(x=jnp.zeros(2), v=jnp.zeros(2))
        ens = solve(_second_order_sde_term(lambda x, v: jnp.zeros_like(v), 0.1),
                    y0, jnp.array(0.0), 10, 1.0, 1.0, EulerHeun(),
                    key=jax.random.key(0), n_samples=5)
        assert ens.x.shape == (5, 11, 2)
        assert ens.v.shape == (5, 11, 2)

    def test_milstein_raises_on_pytree_state(self):
        y0 = State(x=jnp.zeros(2), v=jnp.zeros(2))
        with pytest.raises(NotImplementedError, match="flat array state"):
            solve(_second_order_sde_term(lambda x, v: jnp.zeros_like(v), 0.1),
                  y0, jnp.array(0.0), 5, 1.0, 1.0, ItoMilstein(), key=jax.random.key(0))


class TestLineaxDiffusion:
    def test_matrix_operator_equals_diagonal_when_diagonal(self):
        lx = pytest.importorskip("lineax")
        sigma = 0.2
        y0 = jnp.array([0.0, 0.0])
        key = jax.random.key(3)

        def diag_term(t, y, *args):
            return jnp.zeros(2), jnp.full(2, sigma)

        def op_term(t, y, *args):
            return jnp.zeros(2), lx.DiagonalLinearOperator(jnp.full(2, sigma))

        traj_diag = solve(diag_term, y0, jnp.array(0.0), 10, 1.0, 1.0, EulerHeun(), key=key)
        traj_op = solve(op_term, y0, jnp.array(0.0), 10, 1.0, 1.0, EulerHeun(), key=key)
        assert jnp.allclose(traj_diag, traj_op, atol=1e-6)

    def test_matrix_operator_with_explicit_brownian_structure(self):
        lx = pytest.importorskip("lineax")
        # 2-D state driven by a 3-D Brownian motion via a (2, 3) operator.
        G = jnp.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]) * 1e-2
        y0 = jnp.zeros(2)
        noise_proto = jax.ShapeDtypeStruct((3,), y0.dtype)

        def op_term(t, y, *args):
            return jnp.zeros(2), lx.MatrixLinearOperator(G)

        traj = solve(op_term, y0, jnp.array(0.0), 10, 1.0, 1.0, EulerHeun(),
                     key=jax.random.key(4), brownian_structure=noise_proto)
        assert traj.shape == (11, 2)
        assert jnp.all(jnp.isfinite(traj))
