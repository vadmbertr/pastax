"""Tests for score.py: proper scoring rules for ensemble trajectory forecasts."""

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


class TestDawidSebastiani:
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


class TestVariogramScore:
    def test_shape_default(self):
        f = jnp.ones((5, 7, 2))
        o = jnp.zeros((7, 2))
        assert variogram_score(f, o).shape == (7,)

    def test_reduce_last_is_scalar(self):
        f = jnp.ones((5, 7, 2))
        o = jnp.zeros((7, 2))
        assert variogram_score(f, o, reduce="last").shape == ()

    def test_reduce_sum_matches_sum(self):
        key = jax.random.key(20)
        f = jax.random.normal(key, (5, 7, 2))
        o = jnp.zeros((7, 2))
        full = variogram_score(f, o)
        assert float(variogram_score(f, o, reduce="sum")) == pytest.approx(
            float(full.sum()), rel=1e-5
        )

    def test_reduce_sum_with_weights(self):
        key = jax.random.key(21)
        f = jax.random.normal(key, (5, 7, 2))
        o = jnp.zeros((7, 2))
        w = jnp.arange(7.0) + 1.0
        full = variogram_score(f, o)
        assert float(variogram_score(f, o, reduce="sum", weights=w)) == pytest.approx(
            float((w * full).sum()), rel=1e-5
        )

    def test_hand_value(self):
        # S=2, T=1, p=2, default weights = 1 - I (off-diagonal ones, counted twice).
        # X1=(0,0), X2=(2,4); for sample s, |X_{s,0} - X_{s,1}|^2 is the same off-diagonal value
        # X1: |0 - 0|^2 = 0
        # X2: |2 - 4|^2 = 4
        # E_F = (0 + 4)/2 = 2
        # obs=(1,3): |1 - 3|^2 = 4
        # residual = 2 - 4 = -2; squared = 4; counted twice (i!=j) = 8
        f = jnp.array([[[0.0, 0.0]], [[2.0, 4.0]]])
        o = jnp.array([[1.0, 3.0]])
        assert float(variogram_score(f, o)[0]) == pytest.approx(8.0)

    def test_custom_component_weights(self):
        f = jnp.array([[[0.0, 0.0]], [[2.0, 4.0]]])
        o = jnp.array([[1.0, 3.0]])
        # weight only the (0, 1) entry: half the default score
        w = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        assert float(variogram_score(f, o, component_weights=w)[0]) == pytest.approx(4.0)

    def test_custom_p(self):
        f = jnp.array([[[0.0, 0.0]], [[2.0, 0.0]]])
        o = jnp.array([[0.0, 1.0]])
        # p=1: |X_{s,0} - X_{s,1}|: X1 -> 0, X2 -> 2; E_F = 1
        # obs: |0 - 1| = 1; residual = 0; score = 0
        assert float(variogram_score(f, o, p=1.0)[0]) == pytest.approx(0.0)

    def test_grad_finite(self):
        key = jax.random.key(22)
        f = jax.random.normal(key, (6, 3, 2))
        o = jnp.zeros((3, 2))
        g = jax.grad(lambda a: variogram_score(a, o, reduce="sum"))(f)
        assert jnp.all(jnp.isfinite(g))

    def test_jit_equivalence(self):
        key = jax.random.key(23)
        f = jax.random.normal(key, (6, 3, 2))
        o = jnp.zeros((3, 2))
        a = variogram_score(f, o)
        b = jax.jit(variogram_score)(f, o)
        assert jnp.allclose(a, b)

    @pytest.mark.parametrize("p", [0.5, 1.0])
    def test_grad_finite_for_p_below_two(self, p):
        """|x|^p has an unbounded derivative at x=0 for p < 1; the component
        diagonal is exactly zero, so without the double-where trick the
        backward pass produced nan (0 * inf survives the zero weights)."""
        key = jax.random.key(24)
        f = jax.random.normal(key, (6, 3, 2))
        o = jnp.zeros((3, 2))
        g = jax.grad(lambda a: variogram_score(a, o, p=p, reduce="sum"))(f)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_finite_with_ties_in_forecast(self):
        """Exactly tied ensemble members create off-diagonal zero differences
        too; gradients must stay finite there as well."""
        f = jnp.array([[[1.0, 1.0]], [[1.0, 2.0]]])  # components tied in member 0
        o = jnp.array([[0.0, 1.0]])
        g = jax.grad(lambda a: variogram_score(a, o, p=0.5, reduce="sum"))(f)
        assert jnp.all(jnp.isfinite(g))

    def test_value_unchanged_by_safe_pow(self):
        """The double-where rewrite must not change forward values."""
        f = jax.random.normal(jax.random.key(25), (6, 3, 2))
        o = jax.random.normal(jax.random.key(26), (3, 2))
        for p in (0.5, 1.0, 2.0):
            expected = (
                ((jnp.abs(f[..., :, None] - f[..., None, :]) ** p).mean(0)
                 - jnp.abs(o[..., :, None] - o[..., None, :]) ** p) ** 2
                * (jnp.ones((2, 2)) - jnp.eye(2))
            ).sum(axis=(-2, -1))
            assert jnp.allclose(variogram_score(f, o, p=p), expected, atol=1e-6)


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

    def test_energy_score_two_members_ok(self):
        forecast = jnp.stack([jnp.zeros((5, 2)), jnp.ones((5, 2))])
        obs = jnp.zeros((5, 2))
        assert jnp.all(jnp.isfinite(energy_score(forecast, obs)))

    def test_dawid_sebastiani_two_members_raises(self):
        forecast = jnp.zeros((2, 5, 2))
        obs = jnp.zeros((5, 2))
        with pytest.raises(ValueError, match="S >= 3"):
            dawid_sebastiani(forecast, obs)
