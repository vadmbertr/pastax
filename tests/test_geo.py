"""Tests for geo.py: constants, haversine, unit conversions."""

import jax
import jax.numpy as jnp
import pytest

from pastax.geo import (
    EARTH_RADIUS,
    chordal_projection,
    degrees_to_meters,
    haversine,
    meters_to_degrees,
    wrap_longitude,
)


def test_earth_radius():
    assert EARTH_RADIUS == pytest.approx(6_371_008.8)


class TestChordalProjection:
    def test_analytic_anchors(self):
        points = jnp.array([
            [0.0, 0.0],
            [90.0, 0.0],
            [180.0, 0.0],
            [0.0, 90.0],
            [0.0, -90.0],
        ])
        expected = jnp.array([
            [EARTH_RADIUS, 0.0, 0.0],
            [0.0, EARTH_RADIUS, 0.0],
            [-EARTH_RADIUS, 0.0, 0.0],
            [0.0, 0.0, EARTH_RADIUS],
            [0.0, 0.0, -EARTH_RADIUS],
        ])
        got = chordal_projection(points)
        assert jnp.allclose(got, expected, atol=1e-3)

    def test_unit_sphere_norm(self):
        points = jax.random.uniform(jax.random.PRNGKey(0), (100, 2), minval=-180, maxval=180)
        lats = jax.random.uniform(jax.random.PRNGKey(1), (100,), minval=-90, maxval=90)
        points = points.at[..., 1].set(lats)
        proj = chordal_projection(points)
        norms = jnp.linalg.norm(proj, axis=-1)
        assert jnp.allclose(norms, EARTH_RADIUS, rtol=1e-6)

    def test_shapes_and_broadcasting(self):
        p1 = jnp.array([10.0, 20.0])
        assert chordal_projection(p1).shape == (3,)
        p2 = jnp.zeros((5, 10, 2))
        assert chordal_projection(p2).shape == (5, 10, 3)

    def test_haversine_chord_identity(self):
        p1 = jnp.array([10.0, 20.0])
        p2 = jnp.array([15.0, 25.0])
        h_dist = haversine(p1, p2)
        chord_dist = jnp.linalg.norm(chordal_projection(p1) - chordal_projection(p2))
        expected = 2 * EARTH_RADIUS * jnp.sin(h_dist / (2 * EARTH_RADIUS))
        assert chord_dist == pytest.approx(expected, rel=1e-4)

    def test_round_trip_approx(self):
        p = jnp.array([45.0, 30.0])
        proj = chordal_projection(p)
        p2 = jnp.array([46.0, 31.0])
        assert not jnp.allclose(proj, chordal_projection(p2))

    def test_antimeridian_equivalence(self):
        p1 = jnp.array([180.0, 10.0])
        p2 = jnp.array([-180.0, 10.0])
        assert jnp.allclose(chordal_projection(p1), chordal_projection(p2), atol=1e-3)

    def test_jax_transforms(self):
        p = jnp.array([10.0, 20.0])
        jit_proj = jax.jit(chordal_projection)
        assert jnp.allclose(jit_proj(p), chordal_projection(p))
        pts = jnp.array([[10.0, 20.0], [20.0, 30.0]])
        vmap_proj = jax.vmap(chordal_projection)
        assert jnp.allclose(vmap_proj(pts), chordal_projection(pts))
        grad_fn = jax.grad(lambda x: jnp.sum(chordal_projection(x)**2))
        g = grad_fn(p)
        assert jnp.all(jnp.isfinite(g))


class TestHaversine:

    def test_same_point_is_zero(self):
        y = jnp.array([2.0, 48.0])
        assert float(haversine(y, y)) == pytest.approx(0.0, abs=1e-3)

    def test_north_pole_to_equator(self):
        north = jnp.array([0.0, 90.0])
        equator = jnp.array([0.0, 0.0])
        expected = EARTH_RADIUS * jnp.pi / 2
        assert float(haversine(north, equator)) == pytest.approx(expected, rel=1e-4)

    def test_equator_one_degree_lon(self):
        y1 = jnp.array([0.0, 0.0])
        y2 = jnp.array([1.0, 0.0])
        expected = EARTH_RADIUS * jnp.pi / 180
        assert float(haversine(y1, y2)) == pytest.approx(expected, rel=1e-4)

    def test_grad_finite_at_non_coincident_points(self):
        y1 = jnp.array([2.0, 48.0])
        y2 = jnp.array([3.0, 49.0])
        g = jax.grad(lambda a: haversine(a, y2))(y1)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_finite_at_coincident_points(self):
        y = jnp.array([2.0, 48.0])
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
        assert jnp.allclose(degrees_to_meters(meters_to_degrees(disp_m, lat), lat),
                            disp_m, rtol=1e-5)

    def test_round_trip_at_45deg(self):
        disp_m = jnp.array([5000.0, 5000.0])
        lat = jnp.array(45.0)
        assert jnp.allclose(degrees_to_meters(meters_to_degrees(disp_m, lat), lat),
                            disp_m, rtol=1e-5)

    def test_longitude_scaling_at_60deg(self):
        lat = jnp.array(60.0)
        one_deg = jnp.array([1.0, 1.0])
        m = degrees_to_meters(one_deg, lat)
        assert m[0] == pytest.approx(m[1] * 0.5, rel=1e-3)

    def test_meters_to_degrees_finite_and_safe_at_pole(self):
        """safe_divide keeps the forward value and gradient finite at the pole
        (no inf/nan). The zonal component is still legitimately large there —
        the flat-Earth longitude scaling diverges — but it must not blow up to
        a non-finite value or poison the backward pass."""
        disp_m = jnp.array([1000.0, 1000.0])
        out = meters_to_degrees(disp_m, jnp.array(90.0))
        assert jnp.all(jnp.isfinite(out))
        g = jax.grad(lambda d: meters_to_degrees(d, jnp.array(90.0)).sum())(disp_m)
        assert jnp.all(jnp.isfinite(g))

    def test_meters_to_degrees_unchanged_at_normal_lat(self):
        """The safe_divide swap must not change results where cos(lat) != 0."""
        disp_m = jnp.array([1234.0, -567.0])
        lat = jnp.array(37.0)
        got = meters_to_degrees(disp_m, lat)
        # Independent computation of the same naive formula.
        deg = jnp.degrees(disp_m / EARTH_RADIUS)
        expected = deg.at[0].divide(jnp.cos(jnp.radians(lat)))
        assert jnp.allclose(got, expected, rtol=1e-6)


class TestWrapLongitude:
    def test_default_window_minus180_to_180(self):
        lon = jnp.array([45.0, 181.0, -181.0, 200.0, 540.0])
        got = wrap_longitude(lon)
        # 45->45, 181->-179, -181->179, 200->-160, 540->180->-180
        assert jnp.allclose(got, jnp.array([45.0, -179.0, 179.0, -160.0, -180.0]))

    def test_in_window_values_unchanged(self):
        lon = jnp.array([-180.0, -90.0, 0.0, 90.0, 179.999])
        assert jnp.allclose(wrap_longitude(lon), lon)

    def test_lower_zero_gives_0_360(self):
        lon = jnp.array([-10.0, 10.0, 370.0, 360.0])
        got = wrap_longitude(lon, lower=0.0)
        # -10->350, 10->10, 370->10, 360->0
        assert jnp.allclose(got, jnp.array([350.0, 10.0, 10.0, 0.0]))

    def test_idempotent(self):
        lon = jnp.array([181.0, 540.0, -400.0, 12.3])
        once = wrap_longitude(lon)
        assert jnp.allclose(wrap_longitude(once), once)

    def test_shape_preserved_and_lonlat_column(self):
        # a [lon, lat] trajectory: wrap only the longitude column
        traj = jnp.array([[178.0, 10.0], [181.0, 11.0], [184.0, 12.0]])
        wrapped = traj.at[..., 0].set(wrap_longitude(traj[..., 0]))
        assert wrapped.shape == traj.shape
        assert jnp.allclose(wrapped[:, 0], jnp.array([178.0, -179.0, -176.0]))
        assert jnp.allclose(wrapped[:, 1], traj[:, 1])  # latitude untouched

    def test_jit_vmap_grad_safe(self):
        assert float(jax.jit(wrap_longitude)(jnp.array(181.0))) == pytest.approx(-179.0)
        out = jax.vmap(wrap_longitude)(jnp.array([181.0, -181.0, 200.0]))
        assert jnp.allclose(out, jnp.array([-179.0, 179.0, -160.0]))
        # derivative is 1 away from the wrap seam, and finite
        g = jax.grad(lambda x: wrap_longitude(x))(jnp.array(200.0))
        assert jnp.isfinite(g) and float(g) == pytest.approx(1.0)
