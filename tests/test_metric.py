"""Tests for metrics.py."""

import jax
import jax.numpy as jnp
import pytest

from pastax.geo import EARTH_RADIUS
from pastax.metric import liu_index, normalized_separation_distance, separation_distance


class TestSeparationDistance:
    def test_identical_trajectories_zero(self):
        y = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        assert jnp.allclose(separation_distance(y, y), jnp.zeros(3), atol=1e-3)

    def test_shape(self):
        assert separation_distance(jnp.ones((10, 2)), jnp.zeros((10, 2))).shape == (10,)

    def test_known_distance(self):
        y = jnp.array([[0.0, 0.0]])
        y_ref = jnp.array([[1.0, 0.0]])
        expected = EARTH_RADIUS * jnp.radians(1.0)
        assert float(separation_distance(y, y_ref)[0]) == pytest.approx(expected, rel=1e-4)

    def test_grad_finite(self):
        y_ref = jnp.zeros((5, 2))
        y = jnp.ones((5, 2)) * 0.1
        g = jax.grad(lambda t: separation_distance(t, y_ref).sum())(y)
        assert jnp.all(jnp.isfinite(g))

    def test_ensemble_flag_shape(self):
        ensemble = jnp.ones((4, 10, 2))
        y_ref = jnp.zeros((10, 2))
        dists = separation_distance(ensemble, y_ref)
        assert dists.shape == (4, 10)

    def test_ensemble_consistent_with_manual_vmap(self):
        ensemble = jnp.ones((3, 5, 2)) * 0.1
        y_ref = jnp.zeros((5, 2))
        auto = separation_distance(ensemble, y_ref)
        manual = jax.vmap(lambda y: separation_distance(y, y_ref))(ensemble)
        assert jnp.allclose(auto, manual)


class TestNormalizedSeparationDistance:
    def test_identical_zero(self):
        y = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        nsd = normalized_separation_distance(y, y)
        assert jnp.allclose(nsd, jnp.zeros(3), atol=1e-3)

    def test_shape(self):
        assert normalized_separation_distance(jnp.ones((8, 2)), jnp.zeros((8, 2))).shape == (8,)

    def test_at_t0_result_is_zero(self):
        y = jnp.array([[1.0, 0.0], [2.0, 0.0]])
        y_ref = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        nsd = normalized_separation_distance(y, y_ref)
        assert float(nsd[0]) == pytest.approx(0.0)

    def test_ensemble_flag_shape(self):
        ensemble = jnp.ones((5, 6, 2))
        y_ref = jnp.zeros((6, 2))
        result = normalized_separation_distance(ensemble, y_ref)
        assert result.shape == (5, 6)


class TestLiuIndex:
    def test_identical_zero(self):
        y = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        assert jnp.allclose(liu_index(y, y), jnp.zeros(3), atol=1e-3)

    def test_shape(self):
        assert liu_index(jnp.ones((6, 2)), jnp.zeros((6, 2))).shape == (6,)

    def test_non_decreasing_for_diverging_trajectories(self):
        T = 6
        y = jnp.stack([jnp.arange(T, dtype=float) * 0.1, jnp.zeros(T)], axis=1)
        y_ref = jnp.zeros((T, 2))
        li = liu_index(y, y_ref)
        assert jnp.all(li[2:] >= li[1:-1] - 1e-6)

    def test_grad_finite(self):
        y_ref = jnp.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
        y = jnp.array([[0.05, 0.0], [0.15, 0.0], [0.25, 0.0]])
        g = jax.grad(lambda t: liu_index(t, y_ref).sum())(y)
        assert jnp.all(jnp.isfinite(g))

    def test_ensemble_flag_shape(self):
        ensemble = jnp.ones((3, 7, 2)) * 0.2
        y_ref = jnp.zeros((7, 2))
        result = liu_index(ensemble, y_ref)
        assert result.shape == (3, 7)

    def test_ensemble_consistent_with_manual_vmap(self):
        ensemble = jnp.ones((4, 5, 2)) * 0.1
        y_ref = jnp.zeros((5, 2))
        auto = liu_index(ensemble, y_ref)
        manual = jax.vmap(lambda y: liu_index(y, y_ref))(ensemble)
        assert jnp.allclose(auto, manual)
