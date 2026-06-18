"""Tests for geo.py: constants, haversine, unit conversions."""

import jax
import jax.numpy as jnp
import pytest

from pastax.geo import (
    EARTH_RADIUS,
    degrees_to_meters,
    haversine,
    meters_to_degrees,
)


def test_earth_radius():
    assert EARTH_RADIUS == pytest.approx(6_371_008.8)


class TestHaversine:
    def test_same_point_is_zero(self):
        y = jnp.array([48.0, 2.0])
        assert float(haversine(y, y)) == pytest.approx(0.0, abs=1e-3)

    def test_north_pole_to_equator(self):
        north = jnp.array([90.0, 0.0])
        equator = jnp.array([0.0, 0.0])
        expected = EARTH_RADIUS * jnp.pi / 2
        assert float(haversine(north, equator)) == pytest.approx(expected, rel=1e-4)

    def test_equator_one_degree_lon(self):
        y1 = jnp.array([0.0, 0.0])
        y2 = jnp.array([0.0, 1.0])
        expected = EARTH_RADIUS * jnp.pi / 180
        assert float(haversine(y1, y2)) == pytest.approx(expected, rel=1e-4)

    def test_grad_finite_at_non_coincident_points(self):
        y1 = jnp.array([48.0, 2.0])
        y2 = jnp.array([49.0, 3.0])
        g = jax.grad(lambda a: haversine(a, y2))(y1)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_finite_at_coincident_points(self):
        y = jnp.array([48.0, 2.0])
        g = jax.grad(lambda a: haversine(a, y))(y)
        assert jnp.all(jnp.isfinite(g))

    def test_broadcasts_over_leading_axes(self):
        a = jnp.zeros((3, 4, 2))
        b = jnp.ones((4, 2))
        d = haversine(a, b)
        assert d.shape == (3, 4)
        scalar = haversine(jnp.zeros(2), jnp.ones(2))
        assert jnp.allclose(d, scalar)

    def test_pairwise_broadcast(self):
        pts = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        pairwise = haversine(pts[:, None], pts[None])
        assert pairwise.shape == (3, 3)
        assert jnp.allclose(jnp.diag(pairwise), jnp.zeros(3), atol=1e-3)
        assert jnp.allclose(pairwise, pairwise.T)


class TestUnitConversions:
    def test_round_trip_at_equator(self):
        disp_m = jnp.array([1000.0, 1000.0])
        lat = jnp.array(0.0)
        assert jnp.allclose(degrees_to_meters(meters_to_degrees(disp_m, lat), lat), disp_m, rtol=1e-5)

    def test_round_trip_at_45deg(self):
        disp_m = jnp.array([5000.0, 5000.0])
        lat = jnp.array(45.0)
        assert jnp.allclose(degrees_to_meters(meters_to_degrees(disp_m, lat), lat), disp_m, rtol=1e-5)

    def test_longitude_scaling_at_60deg(self):
        lat = jnp.array(60.0)
        one_deg = jnp.array([1.0, 1.0])
        m = degrees_to_meters(one_deg, lat)
        assert m[1] == pytest.approx(m[0] * 0.5, rel=1e-3)
