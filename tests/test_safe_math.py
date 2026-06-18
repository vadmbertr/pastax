"""Tests for _safe_math.py."""

import jax
import jax.numpy as jnp
import pytest

from pastax._safe_math import safe_divide, safe_log, safe_sqrt


class TestSafeSqrt:
    def test_positive(self):
        assert safe_sqrt(jnp.array(4.0)) == pytest.approx(2.0)

    def test_zero_returns_zero(self):
        assert float(safe_sqrt(jnp.array(0.0))) == 0.0

    def test_negative_returns_zero(self):
        assert float(safe_sqrt(jnp.array(-1.0))) == 0.0

    def test_grad_at_zero_is_finite(self):
        g = jax.grad(lambda x: safe_sqrt(x))(jnp.array(0.0))
        assert jnp.isfinite(g)


class TestSafeLog:
    def test_positive(self):
        assert safe_log(jnp.array(1.0)) == pytest.approx(0.0)

    def test_zero_returns_neginf(self):
        assert jnp.isinf(safe_log(jnp.array(0.0)))

    def test_grad_at_positive_is_finite(self):
        g = jax.grad(lambda x: safe_log(x))(jnp.array(2.0))
        assert jnp.isfinite(g)


class TestSafeDivide:
    def test_normal(self):
        assert safe_divide(jnp.array(6.0), jnp.array(2.0)) == pytest.approx(3.0)

    def test_zero_denom_returns_zero(self):
        assert float(safe_divide(jnp.array(5.0), jnp.array(0.0))) == 0.0

    def test_grad_finite(self):
        g = jax.grad(lambda a: safe_divide(a, jnp.array(2.0)))(jnp.array(3.0))
        assert jnp.isfinite(g)
