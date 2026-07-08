"""Tests for ``Dataset.velocity_interp`` — joint (U, V) interpolation with
optional partial-slip wall correction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pastax import Dataset, Heun, solve


def _agrid_dataset(*, u_const=0.1, v_const=0.0, with_mask=False, land_row=True):
    """Synthetic A-grid dataset with optional land row at lat[0]."""
    nlat, nlon, nt = 5, 10, 2
    lat = np.linspace(0.0, 4.0, nlat).astype(np.float32)
    lon = np.linspace(0.0, 9.0, nlon).astype(np.float32)
    t = np.array([0.0, 1e6], dtype=np.float32)
    u = np.full((nt, nlat, nlon), u_const, dtype=np.float32)
    v = np.full((nt, nlat, nlon), v_const, dtype=np.float32)
    if land_row:
        u[:, 0, :] = 0.0
        v[:, 0, :] = 0.0
    masks = None
    if with_mask:
        m = np.zeros((nlat, nlon), dtype=bool)
        if land_row:
            m[0, :] = True
        masks = {"u": m, "v": m}
    return Dataset.from_arrays(
        {"u": u, "v": v}, t=t, lat=lat, lon=lon, masks=masks,
    )


class TestDefaultScheme:
    """``scheme="default"`` composes per-field ``Field.interp``."""

    def test_default_matches_field_interp_composition(self):
        ds = _agrid_dataset(u_const=0.3, v_const=-0.1, with_mask=False, land_row=False)
        t, lat, lon = jnp.asarray(0.0), jnp.asarray(2.0), jnp.asarray(4.5)
        u, v = ds.velocity_interp(t, lon, lat)
        u_expected = ds["u"].interp(t, lon, lat)
        v_expected = ds["v"].interp(t, lon, lat)
        assert float(u) == pytest.approx(float(u_expected), abs=1e-6)
        assert float(v) == pytest.approx(float(v_expected), abs=1e-6)

    def test_default_returns_u_v_order(self):
        """Returned vector is ``[u, v]`` so it slots into a ``[lon, lat]`` solver
        term as ``[dlon/dt, dlat/dt]``."""
        ds = _agrid_dataset(u_const=1.0, v_const=2.0, with_mask=False, land_row=False)
        result = ds.velocity_interp(jnp.asarray(0.0), jnp.asarray(4.5), jnp.asarray(2.0))
        assert result.shape == (2,)
        assert float(result[0]) == pytest.approx(1.0, abs=1e-6)  # u first
        assert float(result[1]) == pytest.approx(2.0, abs=1e-6)  # v second

    def test_default_with_mask_uses_invdist(self):
        """With a mask present, default scheme uses each Field's invdist."""
        ds = _agrid_dataset(u_const=0.1, v_const=0.0, with_mask=True, land_row=True)
        # Near the coast (lat=0.1): invdist drops the land row entirely.
        u, _ = ds.velocity_interp(jnp.asarray(0.0), jnp.asarray(4.5), jnp.asarray(0.1))
        assert float(u) == pytest.approx(0.1, rel=1e-2)  # full ocean velocity


class TestPartialSlipScheme:
    """``scheme="partialslip"`` joint correction at coast edges."""

    def test_default_a_b_at_coast_gives_half_slip(self):
        """At wl=0 (exactly on coast), partial-slip with a=b=0.5 should
        give 0.5 * U_ocean (the ``slip_a`` value)."""
        ds = _agrid_dataset(u_const=0.1, v_const=0.0, with_mask=True, land_row=True)
        u, v = ds.velocity_interp(
            jnp.asarray(0.0), jnp.asarray(4.5), jnp.asarray(0.0),
            scheme="partialslip",
        )
        # U at coast = (a + b*0) * U_along_N = 0.5 * 0.1
        assert float(u) == pytest.approx(0.05, abs=1e-4)
        # V is 0 everywhere (no longitudinal-coast correction here).
        assert float(v) == pytest.approx(0.0, abs=1e-6)

    def test_free_slip_limit_a1_b0(self):
        """``slip_a=1, slip_b=0`` gives full free-slip — particle sees
        full ocean velocity even at the coast."""
        ds = _agrid_dataset(u_const=0.1, v_const=0.0, with_mask=True, land_row=True)
        u, _ = ds.velocity_interp(
            jnp.asarray(0.0), jnp.asarray(4.5), jnp.asarray(0.0),
            scheme="partialslip", slip_a=1.0, slip_b=0.0,
        )
        assert float(u) == pytest.approx(0.1, abs=1e-4)

    def test_no_slip_limit_a0_b1_matches_naive(self):
        """``slip_a=0, slip_b=1`` gives wl * U_along_N which is exactly
        the naive bilinear (since U_along_S = 0 on the land row)."""
        ds = _agrid_dataset(u_const=0.1, v_const=0.0, with_mask=True, land_row=True)
        # Just above coast — wl=0.25.
        u, _ = ds.velocity_interp(
            jnp.asarray(0.0), jnp.asarray(4.5), jnp.asarray(0.25),
            scheme="partialslip", slip_a=0.0, slip_b=1.0,
        )
        # Naive U = wl * 0.1 = 0.025 (the land row pulls down).
        assert float(u) == pytest.approx(0.025, abs=1e-4)

    def test_partialslip_alongshore_jet_integration(self):
        """End-to-end: a particle near the coast under partial-slip
        ``a=b=0.5`` advects at roughly ``(0.5 + 0.5 * wl) * U_ocean``."""
        ds = _agrid_dataset(u_const=0.1, v_const=0.0, with_mask=True, land_row=True)

        def term(t, y, args):
            uv = args.velocity_interp(t, y[0], y[1], scheme="partialslip")
            return uv  # [u, v] = [dlon/dt, dlat/dt] for a [lon, lat] solver

        # Particle starts at lat=0.1 above a coast (dlat=1 between rows
        # → wl=0.1). Partial-slip factor = 0.5 + 0.5*0.1 = 0.55.
        y0 = jnp.array([0.5, 0.1], dtype=jnp.float32)  # [lon, lat]
        traj = solve(term, y0, jnp.array(0.0), 10, 0.5, 0.5, solver=Heun(), args=ds)
        dlon = float(traj[-1, 0] - traj[0, 0])
        # Expected ≈ 0.55 * 0.1 deg/s * 5 s = 0.275 deg
        assert dlon == pytest.approx(0.275, rel=0.10)
        assert float(traj[-1, 1]) == pytest.approx(0.1, abs=1e-3)  # lat (v) stays 0

    def test_no_edge_fully_land_falls_back_to_naive(self):
        """When no edge is fully land (only a corner is), partial-slip
        leaves U and V at their naive bilinear values."""
        nlat, nlon, nt = 4, 4, 2
        lat = np.linspace(0.0, 3.0, nlat).astype(np.float32)
        lon = np.linspace(0.0, 3.0, nlon).astype(np.float32)
        t = np.array([0.0, 1.0], dtype=np.float32)
        u = np.full((nt, nlat, nlon), 1.0, dtype=np.float32)
        v = np.full((nt, nlat, nlon), 2.0, dtype=np.float32)
        # Only one corner is land at (0, 0):
        u[:, 0, 0] = 0.0
        v[:, 0, 0] = 0.0
        m = np.zeros((nlat, nlon), dtype=bool)
        m[0, 0] = True
        ds = Dataset.from_arrays(
            {"u": u, "v": v}, t=t, lat=lat, lon=lon, masks={"u": m, "v": m},
        )

        # Query inside cell (0..1, 0..1), wl=wj=0.5.
        # No edge is fully land → partial-slip returns naive bilinear:
        #   U = 0.25*0 + 0.25*1 + 0.25*1 + 0.25*1 = 0.75
        #   V = 1.5 similarly
        u_, v_ = ds.velocity_interp(
            jnp.asarray(0.0), jnp.asarray(0.5), jnp.asarray(0.5),
            scheme="partialslip",
        )
        assert float(u_) == pytest.approx(0.75, abs=1e-4)
        assert float(v_) == pytest.approx(1.5,  abs=1e-4)


class TestPartialSlipGradientSafety:
    """``jax.grad`` through ``velocity_interp(scheme='partialslip')`` must
    be NaN-free at all the tricky positions."""

    def _ds(self):
        return _agrid_dataset(u_const=0.1, v_const=0.0, with_mask=True, land_row=True)

    def test_grad_finite_at_coast(self):
        ds = self._ds()
        g = jax.grad(
            lambda p: ds.velocity_interp(
                jnp.asarray(0.0), p[0], p[1], scheme="partialslip"
            ).sum()
        )(jnp.asarray([4.5, 0.0], dtype=jnp.float32))  # [lon, lat] on the coast
        assert jnp.all(jnp.isfinite(g))

    def test_grad_finite_mid_cell(self):
        ds = self._ds()
        g = jax.grad(
            lambda p: ds.velocity_interp(
                jnp.asarray(0.0), p[0], p[1], scheme="partialslip"
            ).sum()
        )(jnp.asarray([4.5, 0.5], dtype=jnp.float32))  # [lon, lat]
        assert jnp.all(jnp.isfinite(g))

    def test_grad_finite_far_from_coast(self):
        ds = self._ds()
        g = jax.grad(
            lambda p: ds.velocity_interp(
                jnp.asarray(0.0), p[0], p[1], scheme="partialslip"
            ).sum()
        )(jnp.asarray([4.5, 3.5], dtype=jnp.float32))  # [lon, lat]
        assert jnp.all(jnp.isfinite(g))


class TestPartialSlipErrors:
    def test_c_grid_raises_not_implemented(self):
        nlat, nlon, nt = 5, 6, 2
        lat = np.linspace(0.0, 4.0, nlat).astype(np.float32)
        lon = np.linspace(0.0, 5.0, nlon).astype(np.float32)
        t = np.array([0.0, 1.0], dtype=np.float32)
        u_data = np.zeros((nt, nlat, nlon - 1), dtype=np.float32)
        v_data = np.zeros((nt, nlat - 1, nlon), dtype=np.float32)
        ds = Dataset.from_arrays_cgrid(
            t, lat, lon,
            {"current": {"u": ("u", u_data), "v": ("v", v_data)}},
        )
        with pytest.raises(NotImplementedError, match="C-grid"):
            ds.velocity_interp(
                jnp.asarray(0.0), jnp.asarray(2.5), jnp.asarray(2.0),
                scheme="partialslip",
            )

    def test_c_grid_fields_raise_even_without_dataset_grid(self):
        """A Dataset built directly (grid=None) around face-staggered fields
        must still be rejected: the A-grid partial-slip maths would read V
        values on U-face coordinates."""
        nlat, nlon, nt = 5, 6, 2
        lat = np.linspace(0.0, 4.0, nlat).astype(np.float32)
        lon = np.linspace(0.0, 5.0, nlon).astype(np.float32)
        t = np.array([0.0, 1.0], dtype=np.float32)
        u_data = np.zeros((nt, nlat, nlon - 1), dtype=np.float32)
        v_data = np.zeros((nt, nlat - 1, nlon), dtype=np.float32)
        cgrid = Dataset.from_arrays_cgrid(
            t, lat, lon,
            {"current": {"u": ("u", u_data), "v": ("v", v_data)}},
        )
        ds = Dataset(fields=cgrid.fields, grid=None)
        with pytest.raises(NotImplementedError, match="C-grid"):
            ds.velocity_interp(
                jnp.asarray(0.0), jnp.asarray(2.5), jnp.asarray(2.0),
                scheme="partialslip",
            )

    def test_partialslip_without_mask_raises(self):
        ds = _agrid_dataset(with_mask=False, land_row=False)
        with pytest.raises(ValueError, match="land mask"):
            ds.velocity_interp(
                jnp.asarray(0.0), jnp.asarray(4.5), jnp.asarray(2.0),
                scheme="partialslip",
            )

    def test_unknown_scheme_raises(self):
        ds = _agrid_dataset()
        with pytest.raises(ValueError, match="Unknown velocity_interp scheme"):
            ds.velocity_interp(
                jnp.asarray(0.0), jnp.asarray(4.5), jnp.asarray(2.0),
                scheme="freeslip",  # type: ignore[arg-type]
            )
