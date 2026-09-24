"""Tests for score.py: proper scoring rules for ensemble trajectory forecasts."""

import warnings

import jax
import jax.numpy as jnp
import pytest

from pastax.geo import haversine
from pastax.score import (
    dawid_sebastiani,
    energy_score,
    l2_distance,
    squared_error,
    variogram_score,
)


def _reference_joint_energy_score(forecast, observation, *, alpha=1.0):
    """User-supplied reference: ES for one joint vector-valued outcome."""
    members = forecast.shape[0]
    if members < 2:
        raise ValueError("joint_energy_score requires at least two ensemble members.")
    observation_distances = l2_distance(forecast, observation) ** alpha
    bias = jnp.mean(observation_distances)
    pairwise_distances = (
        l2_distance(forecast[:, None, :], forecast[None, :, :]) ** alpha
    )
    dispersion = jnp.mean(pairwise_distances) * members / (members - 1)
    return bias - dispersion / 2.0


class TestL2Distance:
    def test_broadcasts(self):
        x = jnp.zeros((4, 3, 2))
        y = jnp.ones((3, 2))
        d = l2_distance(x, y)
        assert d.shape == (4, 3)
        assert jnp.allclose(d, jnp.sqrt(2.0))

    def test_pairwise_broadcast(self):
        x = jnp.arange(6.0).reshape(3, 2)
        pairwise = l2_distance(x[:, None], x[None])
        assert pairwise.shape == (3, 3)
        assert jnp.allclose(jnp.diag(pairwise), jnp.zeros(3))
        assert jnp.allclose(pairwise, pairwise.T)

    def test_grad_finite_at_zero(self):
        x = jnp.array([1.0, 2.0])
        g = jax.grad(lambda a: l2_distance(a, x))(x)
        assert jnp.all(jnp.isfinite(g))
        assert jnp.allclose(g, jnp.zeros_like(g))

    def test_grad_finite_away_from_zero(self):
        x = jnp.array([0.0, 0.0])
        y = jnp.array([3.0, 4.0])
        g = jax.grad(lambda a: l2_distance(a, y))(x)
        assert jnp.all(jnp.isfinite(g))


class TestSquaredError:
    F = jax.random.normal(jax.random.key(202), (5, 4, 2))
    O = jax.random.normal(jax.random.key(203), (4, 2))  # noqa: E741

    def test_shape_default(self):
        f = jnp.ones((5, 7, 2))
        o = jnp.zeros((7, 2))
        assert squared_error(f, o).shape == (7,)

    def test_reduce_last_is_scalar(self):
        f = jnp.ones((5, 7, 2))
        o = jnp.zeros((7, 2))
        s = squared_error(f, o, reduce="last")
        assert s.shape == ()

    def test_reduce_last_matches_index(self):
        f = jnp.ones((5, 7, 2))
        o = jnp.zeros((7, 2))
        full = squared_error(f, o)
        assert float(squared_error(f, o, reduce="last")) == pytest.approx(float(full[-1]))

    def test_reduce_sum_matches_sum(self):
        f = jnp.ones((5, 7, 2))
        o = jnp.zeros((7, 2))
        full = squared_error(f, o)
        assert float(squared_error(f, o, reduce="sum")) == pytest.approx(float(full.sum()))

    def test_reduce_sum_with_weights(self):
        f = jnp.ones((5, 7, 2))
        o = jnp.zeros((7, 2))
        w = jnp.arange(7.0)
        full = squared_error(f, o)
        expected = float((w * full).sum())
        assert float(squared_error(f, o, reduce="sum", weights=w)) == pytest.approx(expected)

    def test_forecast_equals_obs_is_zero(self):
        traj = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        f = jnp.broadcast_to(traj, (4, 2, 2))
        assert jnp.allclose(squared_error(f, traj), jnp.zeros(2))

    def test_hand_value(self):
        f = jnp.array([
            [[1.0, 0.0]],
            [[3.0, 4.0]],
        ])
        o = jnp.array([[0.0, 0.0]])
        # mean = (2, 2); distance to (0, 0) = sqrt(8); squared = 8
        assert float(squared_error(f, o)[0]) == pytest.approx(8.0)

    def test_haversine_kernel_zero(self):
        traj = jnp.array([[2.0, 48.0], [3.0, 49.0]])
        f = jnp.broadcast_to(traj, (3, 2, 2))
        s = squared_error(f, traj, kernel=haversine)
        assert jnp.allclose(s, jnp.zeros(2), atol=1e-3)

    def test_grad_finite(self):
        f = jnp.ones((4, 3, 2)) * 0.5
        o = jnp.zeros((3, 2))
        g = jax.grad(lambda a: squared_error(a, o, reduce="sum"))(f)
        assert jnp.all(jnp.isfinite(g))

    def test_jit_equivalence(self):
        f = jnp.ones((4, 3, 2)) * 0.5
        o = jnp.zeros((3, 2))
        a = squared_error(f, o)
        b = jax.jit(squared_error)(f, o)
        assert jnp.allclose(a, b)

    def test_propriety_smoke(self):
        key = jax.random.key(0)
        o = jnp.zeros((3, 2))
        centered = jax.random.normal(key, (50, 3, 2)) * 0.1
        shifted = centered + jnp.array([1.0, 1.0])
        s_centered = float(squared_error(centered, o, reduce="sum"))
        s_shifted = float(squared_error(shifted, o, reduce="sum"))
        assert s_centered < s_shifted

    def test_reduce_joint_is_scalar(self):
        assert squared_error(self.F, self.O, reduce="joint").shape == ()

    def test_reduce_joint_matches_reference(self):
        expected = l2_distance(self.F.mean(axis=0).reshape(-1), self.O.reshape(-1)) ** 2
        actual = squared_error(self.F, self.O, reduce="joint")
        assert float(actual) == pytest.approx(float(expected), rel=1e-5, abs=1e-6)

    def test_reduce_joint_equals_unweighted_sum_for_l2(self):
        expected = squared_error(self.F, self.O, reduce="sum", weights=None)
        actual = squared_error(self.F, self.O, reduce="joint")
        assert float(actual) == pytest.approx(float(expected), rel=1e-5, abs=1e-6)

    def test_reduce_joint_hand_value(self):
        f = jnp.ones((5, 7, 2))
        o = jnp.zeros((7, 2))
        actual = squared_error(f, o, reduce="joint")
        assert float(actual) == pytest.approx(14.0, rel=1e-5, abs=1e-6)

    def test_reduce_joint_validates_shapes(self):
        with pytest.raises(ValueError, match="matching"):
            squared_error(self.F, jnp.zeros((6, 2)), reduce="joint")

    def test_reduce_joint_ignores_custom_kernel_with_warning(self):
        def raising_kernel(x, y):
            raise RuntimeError("Should not be called")
        expected = squared_error(self.F, self.O, reduce="joint")
        with pytest.warns(UserWarning, match="kernel") as caught:
            actual = squared_error(self.F, self.O, reduce="joint", kernel=raising_kernel)
        assert float(actual) == pytest.approx(float(expected), rel=1e-5, abs=1e-6)
        assert len(caught) == 1

    def test_reduce_joint_no_warning_with_default_kernel(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            squared_error(self.F, self.O, reduce="joint")

    def test_reduce_joint_weights_ignored(self):
        expected = squared_error(self.F, self.O, reduce="joint")
        actual = squared_error(self.F, self.O, reduce="joint", weights=jnp.full((4,), jnp.nan))
        assert float(actual) == pytest.approx(float(expected), rel=1e-5, abs=1e-6)

    def test_reduce_joint_grad_finite(self):
        f = jnp.ones((4, 3, 2)) * 0.5
        o = jnp.zeros((3, 2))
        g = jax.grad(lambda a: squared_error(a, o, reduce="joint"))(f)
        assert jnp.all(jnp.isfinite(g))

    def test_reduce_joint_jit_equivalence(self):
        expected = squared_error(self.F, self.O, reduce="joint")
        actual = jax.jit(lambda f_, o_: squared_error(f_, o_, reduce="joint"))(self.F, self.O)
        assert float(actual) == pytest.approx(float(expected), rel=1e-5, abs=1e-6)


class TestDawidSebastiani:
    def test_reduce_joint_raises(self):
        key = jax.random.key(10)
        f = jax.random.normal(key, (10, 5, 2))
        o = jnp.zeros((5, 2))
        with pytest.raises(ValueError, match="joint"):
            dawid_sebastiani(f, o, reduce="joint")

    def test_reduce_joint_raises_before_size_check(self):
        f = jnp.zeros((2, 5, 2))
        o = jnp.zeros((5, 2))
        with pytest.raises(ValueError, match="joint"):
            dawid_sebastiani(f, o, reduce="joint")

    def test_shape_default(self):
        key = jax.random.key(1)
        f = jax.random.normal(key, (10, 5, 2))
        o = jnp.zeros((5, 2))
        assert dawid_sebastiani(f, o).shape == (5,)

    def test_reduce_last_is_scalar(self):
        key = jax.random.key(2)
        f = jax.random.normal(key, (10, 5, 2))
        o = jnp.zeros((5, 2))
        assert dawid_sebastiani(f, o, reduce="last").shape == ()

    def test_reduce_last_matches_index(self):
        key = jax.random.key(3)
        f = jax.random.normal(key, (10, 5, 2))
        o = jnp.zeros((5, 2))
        full = dawid_sebastiani(f, o)
        assert float(dawid_sebastiani(f, o, reduce="last")) == pytest.approx(float(full[-1]))

    def test_reduce_sum_matches_sum(self):
        key = jax.random.key(4)
        f = jax.random.normal(key, (10, 5, 2))
        o = jnp.zeros((5, 2))
        full = dawid_sebastiani(f, o)
        assert float(dawid_sebastiani(f, o, reduce="sum")) == pytest.approx(
            float(full.sum()), rel=1e-5
        )

    def test_reduce_sum_with_weights(self):
        key = jax.random.key(5)
        f = jax.random.normal(key, (10, 5, 2))
        o = jnp.zeros((5, 2))
        w = jnp.arange(5.0) + 1.0
        full = dawid_sebastiani(f, o)
        assert float(dawid_sebastiani(f, o, reduce="sum", weights=w)) == pytest.approx(
            float((w * full).sum()), rel=1e-5
        )

    def test_minimum_ensemble_size_finite(self):
        key = jax.random.key(6)
        f = jax.random.normal(key, (3, 4, 2))
        o = jnp.zeros((4, 2))
        s = dawid_sebastiani(f, o)
        assert jnp.all(jnp.isfinite(s))

    def test_grad_finite(self):
        key = jax.random.key(7)
        f = jax.random.normal(key, (8, 3, 2))
        o = jnp.zeros((3, 2))
        g = jax.grad(lambda a: dawid_sebastiani(a, o, reduce="sum"))(f)
        assert jnp.all(jnp.isfinite(g))

    def test_jit_equivalence(self):
        key = jax.random.key(8)
        f = jax.random.normal(key, (8, 3, 2))
        o = jnp.zeros((3, 2))
        a = dawid_sebastiani(f, o)
        b = jax.jit(dawid_sebastiani)(f, o)
        assert jnp.allclose(a, b)

    def test_propriety_smoke(self):
        key = jax.random.key(9)
        o = jnp.zeros((3, 2))
        centered = jax.random.normal(key, (100, 3, 2)) * 0.3
        shifted = centered + jnp.array([2.0, 2.0])
        s_centered = float(dawid_sebastiani(centered, o, reduce="sum"))
        s_shifted = float(dawid_sebastiani(shifted, o, reduce="sum"))
        assert s_centered < s_shifted


class TestEnergyScore:
    F = jax.random.normal(jax.random.key(202), (5, 4, 2))
    O = jax.random.normal(jax.random.key(203), (4, 2))  # noqa: E741

    def test_shape_default(self):
        f = jnp.ones((5, 7, 2))
        o = jnp.zeros((7, 2))
        assert energy_score(f, o).shape == (7,)

    def test_reduce_last_is_scalar(self):
        f = jnp.ones((5, 7, 2))
        o = jnp.zeros((7, 2))
        assert energy_score(f, o, reduce="last").shape == ()

    def test_reduce_last_matches_index(self):
        key = jax.random.key(10)
        f = jax.random.normal(key, (5, 7, 2))
        o = jnp.zeros((7, 2))
        full = energy_score(f, o)
        assert float(energy_score(f, o, reduce="last")) == pytest.approx(float(full[-1]))

    def test_reduce_sum_matches_sum(self):
        key = jax.random.key(11)
        f = jax.random.normal(key, (5, 7, 2))
        o = jnp.zeros((7, 2))
        full = energy_score(f, o)
        assert float(energy_score(f, o, reduce="sum")) == pytest.approx(
            float(full.sum()), rel=1e-5
        )

    def test_reduce_sum_with_weights(self):
        key = jax.random.key(12)
        f = jax.random.normal(key, (5, 7, 2))
        o = jnp.zeros((7, 2))
        w = jnp.linspace(0.1, 1.0, 7)
        full = energy_score(f, o)
        assert float(energy_score(f, o, reduce="sum", weights=w)) == pytest.approx(
            float((w * full).sum()), rel=1e-5
        )

    def test_hand_value_unbiased(self):
        # S=2, T=1, alpha=1, L2 kernel.
        # bias term: mean(d(X1,y), d(X2,y))
        # dispersion (unbiased): d(X1, X2) over 1 off-diagonal pair
        # Here: X1=(0,0), X2=(2,0), y=(1,0)
        # bias = (1 + 1) / 2 = 1
        # mean of full 2x2 pairwise: (0 + 2 + 2 + 0)/4 = 1; times S/(S-1) = 2 -> 2
        # score = 1 - 2/2 = 0
        f = jnp.array([[[0.0, 0.0]], [[2.0, 0.0]]])
        o = jnp.array([[1.0, 0.0]])
        assert float(energy_score(f, o)[0]) == pytest.approx(0.0, abs=1e-6)

    def test_dirac_ensemble_reduces_to_distance_alpha(self):
        # All samples identical → dispersion = 0 → ES = d(x, y)^alpha
        x = jnp.array([3.0, 4.0])
        y = jnp.array([0.0, 0.0])
        f = jnp.broadcast_to(x, (6, 1, 2))
        o = y[None]
        assert float(energy_score(f, o, alpha=1.0)[0]) == pytest.approx(5.0)
        assert float(energy_score(f, o, alpha=2.0)[0]) == pytest.approx(25.0)

    def test_haversine_kernel_zero(self):
        traj = jnp.array([[2.0, 48.0], [3.0, 49.0]])
        f = jnp.broadcast_to(traj, (4, 2, 2))
        s = energy_score(f, traj, kernel=haversine)
        assert jnp.allclose(s, jnp.zeros(2), atol=1e-3)

    def test_grad_finite(self):
        key = jax.random.key(13)
        f = jax.random.normal(key, (6, 3, 2))
        o = jnp.zeros((3, 2))
        g = jax.grad(lambda a: energy_score(a, o, reduce="sum"))(f)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_finite_at_dirac_ensemble(self):
        # All samples coincide → diagonal AND off-diagonal of pairwise are zero.
        # Tests that the gradient-safe kernel keeps grads finite.
        x = jnp.array([1.0, 2.0])
        f = jnp.broadcast_to(x, (4, 1, 2))
        o = jnp.array([[0.0, 0.0]])
        g = jax.grad(lambda a: energy_score(a, o, reduce="sum"))(f)
        assert jnp.all(jnp.isfinite(g))

    def test_jit_equivalence(self):
        key = jax.random.key(14)
        f = jax.random.normal(key, (6, 3, 2))
        o = jnp.zeros((3, 2))
        a = energy_score(f, o)
        b = jax.jit(energy_score)(f, o)
        assert jnp.allclose(a, b)

    def test_propriety_smoke(self):
        key = jax.random.key(15)
        o = jnp.zeros((3, 2))
        centered = jax.random.normal(key, (50, 3, 2)) * 0.3
        shifted = centered + jnp.array([1.5, 1.5])
        assert float(energy_score(centered, o, reduce="sum")) < float(
            energy_score(shifted, o, reduce="sum")
        )

    def test_reduce_joint_is_scalar(self):
        assert energy_score(self.F, self.O, reduce="joint").shape == ()

    def test_reduce_joint_matches_reference(self):
        f = jax.random.normal(jax.random.key(200), (6, 5, 2))
        o = jax.random.normal(jax.random.key(201), (5, 2))
        for alpha in (0.5, 1.0, 1.5):
            expected = _reference_joint_energy_score(
                f.reshape(f.shape[0], -1), o.reshape(-1), alpha=alpha
            )
            actual = energy_score(f, o, reduce="joint", alpha=alpha)
            assert float(actual) == pytest.approx(float(expected), rel=1e-5, abs=1e-6)

    def test_reduce_joint_hand_value(self):
        f = jnp.array([[[0.0, 0.0], [0.0, 0.0]], [[2.0, 0.0], [2.0, 0.0]]])
        o = jnp.array([[1.0, 0.0], [0.0, 0.0]])
        expected = (1 + jnp.sqrt(5)) / 2 - jnp.sqrt(2)
        actual = energy_score(f, o, reduce="joint")
        assert float(actual) == pytest.approx(float(expected), rel=1e-5, abs=1e-6)

    def test_reduce_joint_validates_shapes(self):
        with pytest.raises(ValueError, match="matching"):
            energy_score(self.F, jnp.zeros((6, 2)), reduce="joint")
        with pytest.raises(ValueError, match="matching"):
            energy_score(jnp.zeros((4, 2, 3)), jnp.zeros((3, 2)), reduce="joint")
        with pytest.raises(ValueError, match="matching"):
            energy_score(jnp.zeros(6), jnp.zeros(6), reduce="joint")

    def test_reduce_joint_ignores_custom_kernel_with_warning(self):
        def raising_kernel(x, y):
            raise RuntimeError("Should not be called")
        expected = energy_score(self.F, self.O, reduce="joint")
        with pytest.warns(UserWarning, match="kernel") as caught:
            actual = energy_score(self.F, self.O, reduce="joint", kernel=raising_kernel)
        assert float(actual) == pytest.approx(float(expected), rel=1e-5, abs=1e-6)
        assert len(caught) == 1

    def test_reduce_joint_no_warning_with_default_kernel(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            energy_score(self.F, self.O, reduce="joint")

    def test_reduce_joint_weights_ignored(self):
        expected = energy_score(self.F, self.O, reduce="joint")
        actual = energy_score(self.F, self.O, reduce="joint", weights=jnp.full((4,), jnp.nan))
        assert float(actual) == pytest.approx(float(expected), rel=1e-5, abs=1e-6)

    def test_reduce_joint_jit_equivalence(self):
        expected = energy_score(self.F, self.O, reduce="joint")
        actual = jax.jit(lambda f_, o_: energy_score(f_, o_, reduce="joint"))(self.F, self.O)
        assert float(actual) == pytest.approx(float(expected), rel=1e-5, abs=1e-6)

    def test_reduce_joint_warns_at_trace_time_under_jit(self):
        jitted = jax.jit(lambda f_, o_: energy_score(f_, o_, reduce="joint", kernel=haversine))
        with pytest.warns(UserWarning, match="kernel") as caught:
            jitted(self.F, self.O)
        assert len(caught) == 1

    def test_reduce_joint_grad_finite(self):
        f = jax.random.normal(jax.random.key(204), (6, 3, 2))
        o = jnp.zeros((3, 2))
        g = jax.grad(lambda a: energy_score(a, o, reduce="joint"))(f)
        assert jnp.all(jnp.isfinite(g))

    def test_reduce_joint_grad_finite_at_dirac_ensemble(self):
        x = jnp.array([1.0, 2.0])
        f = jnp.broadcast_to(x, (4, 1, 2))
        o = jnp.array([[0.0, 0.0]])
        g = jax.grad(lambda a: energy_score(a, o, reduce="joint"))(f)
        assert jnp.all(jnp.isfinite(g))

    def test_reduce_joint_dirac_value(self):
        x = jnp.array([[3.0, 4.0], [1.0, 2.0]])
        y = jnp.array([[0.0, 0.0], [0.0, 0.0]])
        f = jnp.broadcast_to(x, (6, 2, 2))
        o = y
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        for alpha in (1.0, 2.0):
            expected = l2_distance(x_flat, y_flat) ** alpha
            actual = energy_score(f, o, reduce="joint", alpha=alpha)
            assert float(actual) == pytest.approx(float(expected), rel=1e-5, abs=1e-6)

    def test_reduce_joint_propriety_smoke(self):
        key = jax.random.key(15)
        o = jnp.zeros((3, 2))
        centered = jax.random.normal(key, (50, 3, 2)) * 0.3
        shifted = centered + jnp.array([1.5, 1.5])
        assert float(energy_score(centered, o, reduce="joint")) < float(
            energy_score(shifted, o, reduce="joint")
        )


class TestVariogramScore:
    def test_returns_scalar_and_default_lags(self):
        f = jnp.ones((5, 7, 2))
        o = jnp.zeros((7, 2))
        s1 = variogram_score(f, o)
        s2 = variogram_score(f, o, lags=tuple(range(1, 7)))
        assert s1.shape == ()
        assert jnp.allclose(s1, s2)

    def test_hand_value_nonzero(self):
        f = jnp.array([
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]],
        ])
        o = jnp.array([[0.0, 0.0], [3.0, 0.0], [0.0, 0.0]])
        s = variogram_score(f, o, p=1.0, lags=(1,))
        assert float(s) == pytest.approx(4.0)

    def test_hand_value_calibrated_zero(self):
        f = jnp.array([
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]],
        ])
        o = jnp.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        s = variogram_score(f, o, p=1.0, lags=(1,))
        assert float(s) == pytest.approx(0.0)

    def test_matches_explicit_double_loop(self):
        key = jax.random.key(42)
        f = jax.random.normal(key, (5, 6, 2))
        o = jax.random.normal(jax.random.key(43), (6, 2))
        lags = (1, 2, 4)
        p = 0.5
        
        s_jax = variogram_score(f, o, lags=lags, p=p)
        
        import numpy as np
        f_np = np.array(f, dtype=np.float64)
        o_np = np.array(o, dtype=np.float64)
        S, T, _ = f_np.shape
        
        total_score = 0.0
        for lag in lags:
            lag_score = 0.0
            for t in range(T - lag):
                R = np.zeros(S)
                for s in range(S):
                    diff_f = np.linalg.norm(f_np[s, t+lag] - f_np[s, t])
                    diff_o = np.linalg.norm(o_np[t+lag] - o_np[t])
                    R[s] = diff_f**p - diff_o**p
                
                pair_sum = 0.0
                for s in range(S):
                    for r in range(S):
                        if s != r:
                            pair_sum += R[s] * R[r]
                lag_score += pair_sum / (S * (S - 1))
            total_score += lag_score / (T - lag)
            
        total_score /= len(lags)
        assert float(s_jax) == pytest.approx(total_score, rel=1e-4)

    def test_equal_lag_weights_equal_mean_of_single_lag_scores(self):
        key = jax.random.key(44)
        f = jax.random.normal(key, (5, 6, 2))
        o = jax.random.normal(jax.random.key(45), (6, 2))
        s_both = variogram_score(f, o, lags=(1, 2))
        s_1 = variogram_score(f, o, lags=(1,))
        s_2 = variogram_score(f, o, lags=(2,))
        assert float(s_both) == pytest.approx(float((s_1 + s_2) / 2))

    def test_lag_weights_normalized(self):
        key = jax.random.key(46)
        f = jax.random.normal(key, (5, 6, 2))
        o = jax.random.normal(jax.random.key(47), (6, 2))
        s_default = variogram_score(f, o, lags=(1, 2))
        s_weighted = variogram_score(f, o, lags=(1, 2), lag_weights=jnp.array([2.0, 2.0]))
        assert float(s_default) == pytest.approx(float(s_weighted))

    def test_single_lag_selection(self):
        key = jax.random.key(48)
        f = jax.random.normal(key, (5, 6, 2))
        o = jax.random.normal(jax.random.key(49), (6, 2))
        s_1 = variogram_score(f, o, lags=(1,))
        
        import numpy as np
        f_np = np.array(f, dtype=np.float64)
        o_np = np.array(o, dtype=np.float64)
        S, T, _ = f_np.shape
        p = 0.5
        lag = 1
        lag_score = 0.0
        for t in range(T - lag):
            R = np.zeros(S)
            for s in range(S):
                diff_f = np.linalg.norm(f_np[s, t+lag] - f_np[s, t])
                diff_o = np.linalg.norm(o_np[t+lag] - o_np[t])
                R[s] = diff_f**p - diff_o**p
            pair_sum = 0.0
            for s in range(S):
                for r in range(S):
                    if s != r:
                        pair_sum += R[s] * R[r]
            lag_score += pair_sum / (S * (S - 1))
        lag_score /= (T - lag)
        assert float(s_1) == pytest.approx(lag_score, rel=1e-4)

    def test_perfect_forecast_is_zero(self):
        o = jnp.arange(12.0).reshape(6, 2)
        f = jnp.broadcast_to(o, (4, 6, 2))
        s = variogram_score(f, o)
        assert float(s) == pytest.approx(0.0)

    def test_kernel_hook(self):
        o = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        f = jnp.broadcast_to(o, (4, 3, 2)) + jax.random.normal(jax.random.key(50), (4, 3, 2)) * 0.1
        s_l2 = variogram_score(f, o)
        s_custom = variogram_score(f, o, kernel=lambda x, y: 2 * l2_distance(x, y))
        assert float(s_l2) != float(s_custom)
        
        traj = jnp.array([[2.0, 48.0], [3.0, 49.0], [4.0, 50.0]])
        f_geo = jnp.broadcast_to(traj, (4, 3, 2))
        s_geo = variogram_score(f_geo, traj, kernel=haversine)
        assert float(s_geo) == pytest.approx(0.0, abs=1e-3)

    def test_negative_realized_score_possible(self):
        scores = []
        for seed in range(50):
            key = jax.random.key(seed)
            draws = jax.random.normal(key, (3, 4, 2))
            f = draws[:2]
            o = draws[2]
            scores.append(float(variogram_score(f, o)))
        assert min(scores) < 0.0

    def test_fairness_unbiasedness_smoke(self):
        key = jax.random.key(100)
        
        def single_trial(k, S):
            draws = jax.random.normal(k, (S + 1, 2, 2))
            f = draws[:S]
            o = draws[S]
            
            diff_f = jnp.sqrt(jnp.sum((f[:, 1, :] - f[:, 0, :])**2, axis=-1))
            diff_o = jnp.sqrt(jnp.sum((o[1, :] - o[0, :])**2, axis=-1))
            
            R = diff_f**0.5 - diff_o**0.5
            
            fair = (jnp.sum(R)**2 - jnp.sum(R**2)) / (S * (S - 1))
            naive = jnp.mean(R)**2
            return fair, naive

        keys_pop = jax.random.split(key, 20000)
        pop_fair, pop_naive = jax.vmap(lambda k: single_trial(k, 2))(keys_pop)
        population = float(jnp.mean(pop_fair))
        
        keys_test = jax.random.split(jax.random.key(101), 200)
        test_fair, test_naive = jax.vmap(lambda k: single_trial(k, 4))(keys_test)
        
        mean_fair = float(jnp.mean(test_fair))
        mean_naive = float(jnp.mean(test_naive))
        
        assert abs(mean_fair - population) < 0.25 * population
        assert mean_naive > population

    def test_grad_finite(self):
        key = jax.random.key(102)
        f = jax.random.normal(key, (4, 5, 2))
        o = jax.random.normal(jax.random.key(103), (5, 2))
        g = jax.grad(lambda a: variogram_score(a, o))(f)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_finite_stationary_trajectory(self):
        f = jnp.ones((4, 5, 2))
        o = jnp.ones((5, 2))
        g = jax.grad(lambda a: variogram_score(a, o))(f)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_finite_perfect_forecast(self):
        o = jnp.arange(10.0).reshape(5, 2)
        f = jnp.broadcast_to(o, (4, 5, 2))
        g = jax.grad(lambda a: variogram_score(a, o))(f)
        assert jnp.all(jnp.isfinite(g))

    def test_jit_equivalence(self):
        key = jax.random.key(104)
        f = jax.random.normal(key, (4, 5, 2))
        o = jax.random.normal(jax.random.key(105), (5, 2))
        a = variogram_score(f, o, lags=(1, 2))
        b = jax.jit(lambda x, y: variogram_score(x, y, lags=(1, 2)))(f, o)
        assert jnp.allclose(a, b)

    def test_jit_equivalence_static_argnames(self):
        key = jax.random.key(106)
        f = jax.random.normal(key, (4, 5, 2))
        o = jax.random.normal(jax.random.key(107), (5, 2))
        a = variogram_score(f, o, lags=(1, 2))
        b = jax.jit(variogram_score, static_argnames=("lags",))(f, o, lags=(1, 2))
        assert jnp.allclose(a, b)

    def test_jit_with_traced_lag_weights(self):
        key = jax.random.key(108)
        f = jax.random.normal(key, (4, 5, 2))
        o = jax.random.normal(jax.random.key(109), (5, 2))
        w = jnp.array([1.0, 2.0])
        a = variogram_score(f, o, lags=(1, 2), lag_weights=w)
        b = jax.jit(lambda x, y, weight: variogram_score(x, y, lags=(1, 2), lag_weights=weight))(
            f, o, w
        )
        assert jnp.allclose(a, b)

    def test_jit_with_closure_weights(self):
        key = jax.random.key(110)
        f = jax.random.normal(key, (4, 5, 2))
        o = jax.random.normal(jax.random.key(111), (5, 2))
        w = jnp.array([1.0, 2.0])
        a = variogram_score(f, o, lags=(1, 2), lag_weights=w)
        b = jax.jit(lambda x, y: variogram_score(x, y, lags=(1, 2), lag_weights=w))(f, o)
        assert jnp.allclose(a, b)

    def test_single_member_raises(self):
        f = jnp.zeros((1, 5, 2))
        o = jnp.zeros((5, 2))
        with pytest.raises(ValueError, match="S >= 2"):
            variogram_score(f, o)

    def test_time_mismatch_raises(self):
        f = jnp.zeros((4, 5, 2))
        o = jnp.zeros((6, 2))
        with pytest.raises(ValueError, match="same time dimension"):
            variogram_score(f, o)

    def test_t1_default_lags_raise(self):
        f = jnp.zeros((4, 1, 2))
        o = jnp.zeros((1, 2))
        with pytest.raises(ValueError, match="at least one positive lag"):
            variogram_score(f, o)

    @pytest.mark.parametrize("lags", [(0,), (7,), (8,), (-1,), (1.5,), (True,)])
    def test_invalid_lags_raise(self, lags):
        f = jnp.zeros((4, 7, 2))
        o = jnp.zeros((7, 2))
        with pytest.raises(ValueError):
            variogram_score(f, o, lags=lags)

    def test_invalid_lag_weights_raise(self):
        f = jnp.zeros((4, 5, 2))
        o = jnp.zeros((5, 2))
        with pytest.raises(ValueError):
            variogram_score(f, o, lags=(1, 2), lag_weights=jnp.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError):
            variogram_score(f, o, lags=(1, 2), lag_weights=jnp.array([0.0, 0.0]))
        with pytest.raises(ValueError):
            variogram_score(f, o, lags=(1, 2), lag_weights=jnp.array([-1.0, 2.0]))

def test_reduce_invalid_value_raises():
    f = jnp.ones((3, 2, 2))
    o = jnp.zeros((2, 2))
    with pytest.raises(ValueError, match="reduce"):
        squared_error(f, o, reduce="mean")  # type: ignore[arg-type]


class TestEnsembleSizeValidation:
    """Degenerate ensembles must raise instead of returning inf/NaN."""

    def test_energy_score_single_member_raises(self):
        forecast = jnp.zeros((1, 5, 2))
        obs = jnp.zeros((5, 2))
        with pytest.raises(ValueError, match="S >= 2"):
            energy_score(forecast, obs)

    def test_energy_score_single_member_joint_raises(self):
        forecast = jnp.zeros((1, 5, 2))
        obs = jnp.zeros((5, 2))
        with pytest.raises(ValueError, match="S >= 2"):
            energy_score(forecast, obs, reduce="joint")

    def test_energy_score_two_members_ok(self):
        forecast = jnp.stack([jnp.zeros((5, 2)), jnp.ones((5, 2))])
        obs = jnp.zeros((5, 2))
        assert jnp.all(jnp.isfinite(energy_score(forecast, obs)))

    def test_dawid_sebastiani_two_members_raises(self):
        forecast = jnp.zeros((2, 5, 2))
        obs = jnp.zeros((5, 2))
        with pytest.raises(ValueError, match="S >= 3"):
            dawid_sebastiani(forecast, obs)

    def test_variogram_score_single_member_raises(self):
        forecast = jnp.zeros((1, 5, 2))
        obs = jnp.zeros((5, 2))
        with pytest.raises(ValueError, match="S >= 2"):
            variogram_score(forecast, obs)
