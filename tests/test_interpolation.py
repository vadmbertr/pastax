"""Tests for interpolation.py."""

import jax
import jax.numpy as jnp
import pytest

from pastax.interpolation import (
    _index_and_weight,
    bilinear_interp_2d,
    linear_interp_1d,
    spatiotemporal_interp,
)


class TestIndexAndWeight:
    """Unit tests for the equally-spaced floor-index helper."""

    def test_at_first_node(self):
        coords = jnp.array([0.0, 1.0, 2.0])
        i, w = _index_and_weight(coords, jnp.array(0.0))
        assert int(i) == 0
        assert float(w) == pytest.approx(0.0)

    def test_at_last_node(self):
        coords = jnp.array([0.0, 1.0, 2.0])
        i, w = _index_and_weight(coords, jnp.array(2.0))
        assert int(i) == 1          # clamped to n-2
        assert float(w) == pytest.approx(1.0)

    def test_midpoint(self):
        coords = jnp.array([0.0, 2.0, 4.0])
        i, w = _index_and_weight(coords, jnp.array(1.0))
        assert int(i) == 0
        assert float(w) == pytest.approx(0.5)

    def test_arbitrary_spacing(self):
        coords = jnp.array([10.0, 12.5, 15.0])
        i, w = _index_and_weight(coords, jnp.array(13.75))
        assert int(i) == 1
        assert float(w) == pytest.approx(0.5)

    def test_consistent_with_linear_interp(self):
        # For a linear function values[k] = k, interp should return x exactly
        coords = jnp.linspace(0.0, 4.0, 5)
        values = coords  # values[k] = k
        for xv in [0.3, 1.7, 2.5, 3.9]:
            result = linear_interp_1d(values, coords, jnp.array(xv))
            assert float(result) == pytest.approx(xv, rel=1e-5)


class TestLinearInterp1D:
    def test_at_node(self):
        coords = jnp.array([0.0, 1.0, 2.0])
        values = jnp.array([10.0, 20.0, 30.0])
        assert float(linear_interp_1d(values, coords, jnp.array(1.0))) == pytest.approx(20.0)

    def test_midpoint(self):
        coords = jnp.array([0.0, 2.0])
        values = jnp.array([0.0, 4.0])
        assert float(linear_interp_1d(values, coords, jnp.array(1.0))) == pytest.approx(2.0)

    def test_grad_is_finite(self):
        coords = jnp.array([0.0, 1.0, 2.0])
        values = jnp.array([1.0, 3.0, 6.0])
        g = jax.grad(lambda x: linear_interp_1d(values, coords, x))(jnp.array(0.7))
        assert jnp.isfinite(g)


class TestBilinearInterp2D:
    def setup_method(self):
        # 3x3 grid, values = lat + lon
        self.lats = jnp.array([0.0, 1.0, 2.0])
        self.lons = jnp.array([0.0, 1.0, 2.0])
        self.values = jnp.array(
            [[0.0, 1.0, 2.0],
             [1.0, 2.0, 3.0],
             [2.0, 3.0, 4.0]]
        )

    def test_at_node(self):
        v = bilinear_interp_2d(self.values, self.lats, self.lons, jnp.array(1.0), jnp.array(1.0))
        assert float(v) == pytest.approx(2.0)

    def test_midpoint(self):
        v = bilinear_interp_2d(self.values, self.lats, self.lons, jnp.array(0.5), jnp.array(0.5))
        assert float(v) == pytest.approx(1.0)  # 0.5 + 0.5

    def test_grad_is_finite(self):
        g = jax.grad(
            lambda lat: bilinear_interp_2d(self.values, self.lats, self.lons, lat, jnp.array(0.5))
        )(jnp.array(0.3))
        assert jnp.isfinite(g)

    def test_jit_compatible(self):
        fn = jax.jit(bilinear_interp_2d, static_argnums=())
        v = fn(self.values, self.lats, self.lons, jnp.array(0.5), jnp.array(1.5))
        assert jnp.isfinite(v)


class TestSpatiotemporalInterp:
    def setup_method(self):
        # 2 time steps, 3x3 spatial grid, values = t + lat + lon
        self.t_coords = jnp.array([0.0, 1.0])
        self.lats = jnp.array([0.0, 1.0, 2.0])
        self.lons = jnp.array([0.0, 1.0, 2.0])
        self.values = jnp.stack([
            jnp.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
            jnp.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]),
        ])

    def test_at_node(self):
        v = spatiotemporal_interp(
            self.values, self.t_coords, self.lats, self.lons,
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
        )
        assert float(v) == pytest.approx(0.0)

    def test_mid_time(self):
        v = spatiotemporal_interp(
            self.values, self.t_coords, self.lats, self.lons,
            jnp.array(0.5), jnp.array(1.0), jnp.array(1.0),
        )
        assert float(v) == pytest.approx(2.5)  # 0+1+1=2 at t=0, 1+1+1=3 at t=1 → 2.5

    def test_grad_wrt_lat(self):
        g = jax.grad(
            lambda lat: spatiotemporal_interp(
                self.values, self.t_coords, self.lats, self.lons,
                jnp.array(0.5), lat, jnp.array(1.0),
            )
        )(jnp.array(0.5))
        assert jnp.isfinite(g)


class TestBilinearInterp2DLonPeriodic:
    """Longitude wrap-around (360°→0° discontinuity) in bilinear_interp_2d."""

    def setup_method(self):
        # n_lon = 4, dx = 90°, grid = [0, 90, 180, 270] spans one full period
        self.lats = jnp.array([0.0, 1.0])
        self.lons = jnp.array([0.0, 90.0, 180.0, 270.0])
        # values vary only with lon: values[:, j] = j (so we see indices directly)
        self.values = jnp.array([[0.0, 1.0, 2.0, 3.0],
                                 [0.0, 1.0, 2.0, 3.0]])

    def test_at_node_zero(self):
        v = bilinear_interp_2d(
            self.values, self.lats, self.lons,
            jnp.array(0.0), jnp.array(0.0), lon_period=360.0,
        )
        assert float(v) == pytest.approx(0.0)

    def test_wraps_between_last_and_first(self):
        # halfway between 270° (index 3) and 360° == 0° (index 0): (3 + 0) / 2
        v = bilinear_interp_2d(
            self.values, self.lats, self.lons,
            jnp.array(0.0), jnp.array(315.0), lon_period=360.0,
        )
        assert float(v) == pytest.approx(1.5)

    def test_negative_lon_wraps(self):
        # -45° == 315° → same midpoint between 270 and 0
        v = bilinear_interp_2d(
            self.values, self.lats, self.lons,
            jnp.array(0.0), jnp.array(-45.0), lon_period=360.0,
        )
        assert float(v) == pytest.approx(1.5)

    def test_lon_above_period_wraps(self):
        # 405° == 45° → midpoint between index 0 and index 1
        v = bilinear_interp_2d(
            self.values, self.lats, self.lons,
            jnp.array(0.0), jnp.array(405.0), lon_period=360.0,
        )
        assert float(v) == pytest.approx(0.5)

    def test_no_lon_period_does_not_wrap(self):
        # Without lon_period, querying at 315° extrapolates from cell [2,3]
        v = bilinear_interp_2d(
            self.values, self.lats, self.lons,
            jnp.array(0.0), jnp.array(315.0),
        )
        # cell [j=2..3]: values 2, 3 → at 315°, w = (315 - 180)/90 = 1.5 → 2*(1-1.5)+3*1.5 = 3.5
        assert float(v) == pytest.approx(3.5)

    def test_grad_finite_across_seam(self):
        g = jax.grad(
            lambda lon: bilinear_interp_2d(
                self.values, self.lats, self.lons,
                jnp.array(0.0), lon, lon_period=360.0,
            )
        )(jnp.array(355.0))
        assert jnp.isfinite(g)

    def test_jit_compatible_periodic(self):
        @jax.jit
        def f(lon):
            return bilinear_interp_2d(
                self.values, self.lats, self.lons,
                jnp.array(0.0), lon, lon_period=360.0,
            )
        v = f(jnp.array(315.0))
        assert float(v) == pytest.approx(1.5)


class TestSpatiotemporalInterpLonPeriodic:
    """Longitude wrap-around in spatiotemporal_interp."""

    def test_wraps_in_time_and_space(self):
        t_coords = jnp.array([0.0, 1.0])
        lats = jnp.array([0.0, 1.0])
        lons = jnp.array([0.0, 90.0, 180.0, 270.0])
        # values[t, lat, lon] = lon_index (constant in t, lat)
        slab = jnp.broadcast_to(jnp.array([0.0, 1.0, 2.0, 3.0]), (2, 4))
        values = jnp.stack([slab, slab])  # (2, 2, 4)
        v = spatiotemporal_interp(
            values, t_coords, lats, lons,
            jnp.array(0.5), jnp.array(0.0), jnp.array(315.0),
            lon_period=360.0,
        )
        assert float(v) == pytest.approx(1.5)
