"""Tests for the optional land-mask plumbing on Field and the loaders.

- ``Field.mask`` defaults to ``None`` (backwards-compatible PyTree).
- All four loaders accept a ``masks`` kwarg.
- When the source contains NaN, a mask is auto-inferred and NaN is
  replaced with 0 in the stored ``values``.
- When the source is NaN-free and no user mask is supplied, the
  resulting ``Field.mask`` stays ``None``.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from pastax import Dataset, Field


def _uv_vectors(u, v, group="current", u_name="u", v_name="v"):
    """Wrap a single (u, v) pair into the new C-grid `vectors` dict shape."""
    return {group: {"u": (u_name, u), "v": (v_name, v)}}


def test_field_mask_defaults_to_none():
    f = Field.standalone(
        values=jnp.zeros((2, 3, 4)),
        t_coords=jnp.asarray([0.0, 1.0]),
        lat_coords=jnp.linspace(0.0, 2.0, 3),
        lon_coords=jnp.linspace(0.0, 3.0, 4),
    )
    assert f.mask is None


@pytest.mark.parametrize("stagger", ["u_face", "v_face"])
def test_standalone_rejects_lon_period_on_faces(stagger):
    """Face fields never wrap in longitude; lon_period used to be silently
    dropped for face staggers — it must be rejected loudly instead."""
    with pytest.raises(ValueError, match="lon_period is not supported"):
        Field.standalone(
            values=jnp.zeros((2, 3, 4)),
            t_coords=jnp.asarray([0.0, 1.0]),
            lat_coords=jnp.linspace(0.0, 2.0, 3),
            lon_coords=jnp.linspace(0.0, 3.0, 4),
            lon_period=4.0,
            stagger=stagger,
        )


def test_field_accepts_explicit_mask():
    mask = jnp.array([[False, True], [True, False]])
    f = Field.standalone(
        values=jnp.zeros((1, 2, 2)),
        t_coords=jnp.asarray([0.0]),
        lat_coords=jnp.asarray([0.0, 1.0]),
        lon_coords=jnp.asarray([0.0, 1.0]),
        mask=mask,
    )
    assert f.mask is not None
    assert jnp.array_equal(f.mask, mask)


class TestFromArraysMask:
    def _coords(self):
        t = np.linspace(0.0, 7200.0, 3)
        lat = np.linspace(0.0, 3.0, 4)
        lon = np.linspace(10.0, 14.0, 5)
        return t, lat, lon

    def test_nan_free_input_leaves_mask_none(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 4, 5), dtype=np.float32)
        ds = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
        assert ds["u"].mask is None

    def test_nan_input_auto_infers_mask(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 4, 5), dtype=np.float32)
        u[:, 0, 0] = np.nan       # one land cell
        u[1, 2, 3] = np.nan       # NaN at one timestep only — masked, with a warning
        with pytest.warns(UserWarning, match="some but not all time steps"):
            ds = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)

        assert ds["u"].mask is not None
        assert ds["u"].mask.shape == (4, 5)
        assert bool(ds["u"].mask[0, 0]) is True
        assert bool(ds["u"].mask[2, 3]) is True
        # An ocean cell stays False
        assert bool(ds["u"].mask[1, 1]) is False

    def test_nan_values_replaced_with_zero(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 4, 5), dtype=np.float32)
        u[:, 0, 0] = np.nan
        ds = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
        assert bool(jnp.isnan(ds["u"].values).any()) is False
        assert float(ds["u"].values[0, 0, 0]) == 0.0

    def test_user_mask_overrides_inference(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 4, 5), dtype=np.float32)
        u[:, 0, 0] = np.nan       # would be inferred as land
        explicit = np.zeros((4, 5), dtype=bool)
        explicit[3, 4] = True     # different cell than the NaN
        ds = Dataset.from_arrays(
            {"u": u}, t=t, lat=lat, lon=lon, masks={"u": explicit},
        )
        # User mask wins; the NaN-derived cell is NOT marked as land.
        assert bool(ds["u"].mask[0, 0]) is False
        assert bool(ds["u"].mask[3, 4]) is True
        # NaN was still cleared from values regardless.
        assert bool(jnp.isnan(ds["u"].values).any()) is False

    def test_per_field_mask_routing(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 4, 5), dtype=np.float32)
        v = np.ones((3, 4, 5), dtype=np.float32)
        u_mask = np.zeros((4, 5), dtype=bool)
        u_mask[0, 0] = True
        ds = Dataset.from_arrays(
            {"u": u, "v": v}, t=t, lat=lat, lon=lon, masks={"u": u_mask},
        )
        # u gets the explicit mask, v stays None (no NaN, no user mask).
        assert ds["u"].mask is not None and bool(ds["u"].mask[0, 0]) is True
        assert ds["v"].mask is None

    def test_rejects_3d_mask(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 4, 5), dtype=np.float32)
        bad = np.zeros((3, 4, 5), dtype=bool)
        with pytest.raises(ValueError, match=r"expected 2-D bool array"):
            Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon, masks={"u": bad})

    def test_rejects_wrong_2d_shape(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 4, 5), dtype=np.float32)
        bad = np.zeros((4, 4), dtype=bool)
        with pytest.raises(ValueError, match=r"expected 2-D bool array"):
            Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon, masks={"u": bad})


class TestFromXarrayMask:
    def _ds_with_nan_land(self):
        t = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
        lat = np.linspace(0.0, 2.0, 3)
        lon = np.linspace(10.0, 13.0, 4)
        u = np.ones((2, 3, 4), dtype=np.float32)
        u[:, 0, 0] = np.nan
        return xr.Dataset(
            {"u": (["time", "lat", "lon"], u)},
            coords={"time": t, "lat": lat, "lon": lon},
        )

    def test_nan_in_xarray_auto_infers_mask(self):
        ds_xr = self._ds_with_nan_land()
        ds = Dataset.from_xarray(
            ds_xr,
            fields={"u": "u"},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        assert ds["u"].mask is not None
        assert bool(ds["u"].mask[0, 0]) is True
        assert bool(jnp.isnan(ds["u"].values).any()) is False

    def test_explicit_mask_via_from_xarray(self):
        ds_xr = self._ds_with_nan_land()
        explicit = np.zeros((3, 4), dtype=bool)
        explicit[2, 3] = True
        ds = Dataset.from_xarray(
            ds_xr,
            fields={"u": "u"},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
            masks={"u": explicit},
        )
        # User mask wins over NaN inference.
        assert bool(ds["u"].mask[0, 0]) is False
        assert bool(ds["u"].mask[2, 3]) is True


class TestFromArraysCGridMask:
    def _inputs(self, nlat=5, nlon=6, nt=2):
        t = np.linspace(0.0, 3600.0, nt)
        lat = np.linspace(0.0, float(nlat - 1), nlat)
        lon = np.linspace(0.0, float(nlon - 1), nlon)
        u = np.ones((nt, nlat, nlon - 1), dtype=np.float32)
        v = np.ones((nt, nlat - 1, nlon), dtype=np.float32)
        return t, lat, lon, u, v

    def test_nan_in_u_infers_u_face_mask(self):
        t, lat, lon, u, v = self._inputs()
        u[:, 0, 0] = np.nan
        ds = Dataset.from_arrays_cgrid(t, lat, lon, _uv_vectors(u, v))
        assert ds["u"].mask is not None
        assert ds["u"].mask.shape == (5, 5)  # (nlat, nlon - 1)
        assert bool(ds["u"].mask[0, 0]) is True
        assert ds["v"].mask is None

    def test_nan_in_v_infers_v_face_mask(self):
        t, lat, lon, u, v = self._inputs()
        v[:, 0, 0] = np.nan
        ds = Dataset.from_arrays_cgrid(t, lat, lon, _uv_vectors(u, v))
        assert ds["u"].mask is None
        assert ds["v"].mask is not None
        assert ds["v"].mask.shape == (4, 6)  # (nlat - 1, nlon)

    def test_nan_in_tracer_infers_centre_mask(self):
        t, lat, lon, u, v = self._inputs()
        sst = np.full((2, 5, 6), 15.0, dtype=np.float32)
        sst[:, 0, 0] = np.nan
        ds = Dataset.from_arrays_cgrid(
            t, lat, lon, _uv_vectors(u, v), tracers={"sst": sst},
        )
        assert ds["sst"].mask is not None
        assert ds["sst"].mask.shape == (5, 6)
        assert bool(ds["sst"].mask[0, 0]) is True

    def test_explicit_u_face_mask_with_correct_shape(self):
        t, lat, lon, u, v = self._inputs()
        u_mask = np.zeros((5, 5), dtype=bool)
        u_mask[2, 3] = True
        ds = Dataset.from_arrays_cgrid(
            t, lat, lon, _uv_vectors(u, v), masks={"u": u_mask},
        )
        assert bool(ds["u"].mask[2, 3]) is True

    def test_explicit_u_mask_wrong_shape_rejected(self):
        t, lat, lon, u, v = self._inputs()
        # Centre shape instead of u-face shape
        wrong = np.zeros((5, 6), dtype=bool)
        with pytest.raises(ValueError, match=r"masks\['u'\]"):
            Dataset.from_arrays_cgrid(
                t, lat, lon, _uv_vectors(u, v), masks={"u": wrong},
            )

    def test_explicit_v_mask_wrong_shape_rejected(self):
        t, lat, lon, u, v = self._inputs()
        wrong = np.zeros((5, 6), dtype=bool)
        with pytest.raises(ValueError, match=r"masks\['v'\]"):
            Dataset.from_arrays_cgrid(
                t, lat, lon, _uv_vectors(u, v), masks={"v": wrong},
            )

    def test_explicit_tracer_mask_routed_correctly(self):
        t, lat, lon, u, v = self._inputs()
        sst = np.full((2, 5, 6), 15.0, dtype=np.float32)
        tr_mask = np.zeros((5, 6), dtype=bool)
        tr_mask[1, 1] = True
        ds = Dataset.from_arrays_cgrid(
            t, lat, lon, _uv_vectors(u, v),
            tracers={"sst": sst}, masks={"sst": tr_mask},
        )
        assert bool(ds["sst"].mask[1, 1]) is True
        # u/v still mask-less (no NaN, no user mask)
        assert ds["u"].mask is None
        assert ds["v"].mask is None

    def test_masks_routed_to_renamed_vector_fields(self):
        """Masks keyed by user-supplied field names should reach the
        right Field even when those names are not 'u' / 'v'."""
        t, lat, lon, u, v = self._inputs()
        uo_mask = np.zeros((5, 5), dtype=bool)
        uo_mask[1, 2] = True
        u10_mask = np.zeros((5, 5), dtype=bool)
        u10_mask[3, 4] = True
        u_wind = u.copy()
        v_wind = v.copy()
        ds = Dataset.from_arrays_cgrid(
            t, lat, lon,
            {
                "current": {"u": ("uo",  u),      "v": ("vo",  v)},
                "wind":    {"u": ("u10", u_wind), "v": ("v10", v_wind)},
            },
            masks={"uo": uo_mask, "u10": u10_mask},
        )
        assert bool(ds["uo"].mask[1, 2]) is True
        assert bool(ds["u10"].mask[3, 4]) is True
        # Untouched V fields stay mask-less.
        assert ds["vo"].mask is None
        assert ds["v10"].mask is None


class TestFromXarrayCGridMask:
    def test_nan_in_xarray_cgrid_infers_mask(self):
        t = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
        lat = np.linspace(0.0, 4.0, 5)
        lon = np.linspace(0.0, 5.0, 6)
        u_data = np.ones((2, 5, 5), dtype=np.float32)
        v_data = np.ones((2, 4, 6), dtype=np.float32)
        u_data[:, 0, 0] = np.nan
        ds_xr = xr.Dataset(
            {
                "uo": (["time", "lat", "lon_u"], u_data),
                "vo": (["time", "lat_v", "lon"], v_data),
            },
            coords={
                "time": t, "lat": lat, "lon": lon,
                "lon_u": lon[:-1] + 0.5 * (lon[1] - lon[0]),
                "lat_v": lat[:-1] + 0.5 * (lat[1] - lat[0]),
            },
        )
        ds = Dataset.from_xarray_cgrid(
            ds_xr,
            vectors={"current": {"u": ("u", "uo"), "v": ("v", "vo")}},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        assert ds["u"].mask is not None
        assert bool(ds["u"].mask[0, 0]) is True
        assert ds["v"].mask is None
        assert bool(jnp.isnan(ds["u"].values).any()) is False


def test_interp_unchanged_when_mask_none():
    """For NaN-free input the mask path is fully skipped."""
    t = np.linspace(0.0, 3600.0, 3)
    lat = np.linspace(0.0, 4.0, 5)
    lon = np.linspace(0.0, 5.0, 6)
    rng = np.random.default_rng(0)
    u = rng.standard_normal((3, 5, 6)).astype(np.float32)

    ds = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
    assert ds["u"].mask is None
    val = float(ds["u"].interp(jnp.asarray(1800.0), jnp.asarray(3.1), jnp.asarray(2.3)))
    # Build a Field directly without going through the loader — should match.
    f_direct = Field.standalone(
        values=jnp.asarray(u),
        t_coords=jnp.asarray(t, dtype=jnp.float32),
        lat_coords=jnp.asarray(lat, dtype=jnp.float32),
        lon_coords=jnp.asarray(lon, dtype=jnp.float32),
    )
    val_direct = float(
        f_direct.interp(jnp.asarray(1800.0), jnp.asarray(3.1), jnp.asarray(2.3))
    )
    assert val == pytest.approx(val_direct, abs=1e-6)


class TestMaskedBilinear:
    """``Field.interp`` with a mask present: invdist on coastal cells,
    zero on fully-land cells, bilinear-identical when all four corners are
    ocean."""

    def _field(self, values: np.ndarray, mask: np.ndarray) -> Field:
        nlat, nlon = values.shape[-2:]
        return Field.standalone(
            values=jnp.asarray(values, dtype=jnp.float32),
            t_coords=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            lat_coords=jnp.linspace(0.0, float(nlat - 1), nlat, dtype=jnp.float32),
            lon_coords=jnp.linspace(0.0, float(nlon - 1), nlon, dtype=jnp.float32),
            mask=jnp.asarray(mask, dtype=jnp.bool_),
        )

    def test_all_ocean_cell_matches_naive(self):
        """When all four corners are ocean, masked interp must match
        unmasked bilinear bit-for-bit (modulo float32)."""
        values = np.array(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]] * 2,
            dtype=np.float32,
        )
        mask_all_ocean = np.zeros((3, 3), dtype=bool)
        f_masked = self._field(values, mask_all_ocean)
        f_naive = Field.standalone(
            values=jnp.asarray(values),
            t_coords=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            lat_coords=jnp.linspace(0.0, 2.0, 3, dtype=jnp.float32),
            lon_coords=jnp.linspace(0.0, 2.0, 3, dtype=jnp.float32),
        )
        for lat_q, lon_q in [(0.3, 0.7), (1.5, 1.5), (1.9, 0.1)]:
            v_m = float(f_masked.interp(jnp.asarray(0.5), jnp.asarray(lon_q), jnp.asarray(lat_q)))
            v_n = float(f_naive.interp(jnp.asarray(0.5), jnp.asarray(lon_q), jnp.asarray(lat_q)))
            assert v_m == pytest.approx(v_n, abs=1e-6)

    def test_all_land_cell_returns_zero(self):
        """A query inside a 2×2 sub-cell where all four corners are land
        must return exactly 0 (no NaN, no extrapolation)."""
        values = np.ones((2, 3, 3), dtype=np.float32) * 5.0
        mask = np.ones((3, 3), dtype=bool)  # everything is land
        f = self._field(values, mask)
        v = float(f.interp(jnp.asarray(0.5), jnp.asarray(1.5), jnp.asarray(1.5)))
        assert v == 0.0

    def test_one_land_corner_uses_only_ocean(self):
        """One land corner: result should equal an inverse-distance weighted
        average of the three ocean corners' values, NOT include the land
        corner's stored value at all."""
        # Layout (lat × lon, integer indices):
        #   (0,0)=ocean=10   (0,1)=ocean=20
        #   (1,0)=LAND=999   (1,1)=ocean=40
        slab = np.array([[10.0, 20.0], [999.0, 40.0]], dtype=np.float32)
        values = np.stack([slab, slab])
        mask = np.array([[False, False], [True, False]], dtype=bool)
        f = self._field(values, mask)

        # Query at the centre of the cell: equal cell-coord distance to each
        # of the four corners (wl=wj=0.5). Drop (1,0); average the other
        # three with equal inverse-distance weight → (10+20+40)/3 = 23.33...
        v_centre = float(f.interp(jnp.asarray(0.5), jnp.asarray(0.5), jnp.asarray(0.5)))
        assert v_centre == pytest.approx((10.0 + 20.0 + 40.0) / 3.0, rel=1e-5)
        # Crucially, 999 was NOT included.
        assert v_centre < 100.0

    def test_query_at_ocean_corner_recovers_corner_value(self):
        """A query exactly on an ocean corner of a mixed-mask cell must
        approach that corner's value (the eps floor keeps it finite)."""
        slab = np.array([[10.0, 20.0], [0.0, 40.0]], dtype=np.float32)
        values = np.stack([slab, slab])
        mask = np.array([[False, False], [True, False]], dtype=bool)
        f = self._field(values, mask)
        v = float(f.interp(jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0)))
        assert v == pytest.approx(10.0, rel=1e-4)

    def test_query_at_land_corner_uses_distant_ocean(self):
        """A query at a land corner: inverse-distance weighting kicks in
        because the eps floor prevents the would-be ∞ weight at the land
        corner from dominating (the land corner is masked out entirely).
        Result must be finite, no NaN."""
        slab = np.array([[10.0, 20.0], [0.0, 40.0]], dtype=np.float32)
        values = np.stack([slab, slab])
        mask = np.array([[False, False], [True, False]], dtype=bool)
        f = self._field(values, mask)
        v = float(f.interp(jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1.0)))
        assert np.isfinite(v)


class TestMaskedInterpGradientSafety:
    """Backward-pass safety: ``jax.grad`` through ``Field.interp`` must
    produce no NaN at any of the tricky positions (corners, on land,
    fully-land cell)."""

    def _setup(self):
        slab = np.array([[10.0, 20.0], [0.0, 40.0]], dtype=np.float32)
        values = np.stack([slab, slab])
        mask = np.array([[False, False], [True, False]], dtype=bool)
        return Field.standalone(
            values=jnp.asarray(values),
            t_coords=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            lat_coords=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            lon_coords=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            mask=jnp.asarray(mask),
        )

    def test_grad_finite_in_mixed_cell(self):
        import jax
        f = self._setup()
        g = jax.grad(
            lambda p: f.interp(jnp.asarray(0.5), p[0], p[1])
        )(jnp.asarray([0.7, 0.3], dtype=jnp.float32))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_finite_at_ocean_corner(self):
        import jax
        f = self._setup()
        g = jax.grad(
            lambda p: f.interp(jnp.asarray(0.5), p[0], p[1])
        )(jnp.asarray([0.0, 0.0], dtype=jnp.float32))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_finite_at_land_corner(self):
        import jax
        f = self._setup()
        g = jax.grad(
            lambda p: f.interp(jnp.asarray(0.5), p[0], p[1])
        )(jnp.asarray([0.0, 1.0], dtype=jnp.float32))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_finite_in_all_land_cell(self):
        import jax
        values = np.ones((2, 2, 2), dtype=np.float32) * 5.0
        mask = np.ones((2, 2), dtype=bool)
        f = Field.standalone(
            values=jnp.asarray(values),
            t_coords=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            lat_coords=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            lon_coords=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            mask=jnp.asarray(mask),
        )
        g = jax.grad(
            lambda p: f.interp(jnp.asarray(0.5), p[0], p[1])
        )(jnp.asarray([0.5, 0.5], dtype=jnp.float32))
        assert jnp.all(jnp.isfinite(g))


class TestAlongshoreJetNoStuckParticle:
    """End-to-end coastal-robustness check: an A-grid alongshore jet must
    advect a particle along the coast — not trap it.

    Setup: lat∈[0, 4], lon∈[0, 9]. The first row (lat[0]) is LAND; the
    rest is ocean. U is a constant 0.1 deg/s eastward; V is zero. Without
    a mask, the bilinear interp pulls the particle east-velocity towards
    0 as the particle gets near lat=0, because the land row carries zero
    U. With the mask + invdist, the land row is dropped and the particle
    keeps the full ocean velocity.
    """

    def _dataset(self, with_mask: bool):
        from pastax import Dataset
        nlat, nlon, nt = 5, 10, 2
        lat = np.linspace(0.0, 4.0, nlat).astype(np.float32)
        lon = np.linspace(0.0, 9.0, nlon).astype(np.float32)
        t = np.array([0.0, 1e6], dtype=np.float32)
        u = np.full((nt, nlat, nlon), 0.1, dtype=np.float32)
        u[:, 0, :] = 0.0  # land row carries zero velocity
        v = np.zeros((nt, nlat, nlon), dtype=np.float32)
        masks = None
        if with_mask:
            m = np.zeros((nlat, nlon), dtype=bool)
            m[0, :] = True
            masks = {"u": m, "v": m}
        return Dataset.from_arrays(
            {"u": u, "v": v}, t=t, lat=lat, lon=lon, masks=masks,
        )

    def _term(self):
        # Treat U/V as deg/s directly so the test isolates the masking
        # logic from the meters_to_degrees conversion.
        def term(t, y, args):
            ds = args
            u = ds["u"].interp(t, y[0], y[1])
            v = ds["v"].interp(t, y[0], y[1])
            return jnp.array([u, v])
        return term

    def test_unmasked_particle_gets_stuck_near_coast(self):
        from pastax import Heun, solve
        ds = self._dataset(with_mask=False)
        y0 = jnp.array([0.5, 0.1], dtype=jnp.float32)  # [lon, lat], just above coast
        traj = solve(self._term(), y0, jnp.array(0.0), 10, 0.5, 0.5,
                     solver=Heun(), args=ds)  # 5 seconds
        # With naive bilinear, the lat=0 land row pulls U down: at lat=0.1
        # (close to coast) U≈0.1*0.1≈0.01 deg/s, so dlon ≈ 0.05 over 5 s.
        dlon_unmasked = float(traj[-1, 0] - traj[0, 0])
        assert dlon_unmasked < 0.10  # well under the full 0.5 deg eastward

    def test_masked_particle_slides_along_coast(self):
        from pastax import Heun, solve
        ds = self._dataset(with_mask=True)
        y0 = jnp.array([0.5, 0.1], dtype=jnp.float32)  # [lon, lat]
        traj = solve(self._term(), y0, jnp.array(0.0), 10, 0.5, 0.5, solver=Heun(), args=ds)
        # With the mask + invdist, the land row is dropped: the particle
        # sees the ocean U=0.1 unattenuated. Over 5 s → 0.5 deg east.
        dlon_masked = float(traj[-1, 0] - traj[0, 0])
        assert dlon_masked == pytest.approx(0.5, rel=0.05)
        # And lat should not drift (V=0 everywhere).
        assert float(traj[-1, 1]) == pytest.approx(0.1, abs=1e-3)


class TestClosedBay:
    """A particle inside a cell entirely surrounded by land must have
    zero velocity (not NaN) and stay put."""

    def test_closed_bay_zero_velocity(self):
        from pastax import Dataset, Heun, solve
        nlat, nlon, nt = 5, 5, 2
        lat = np.linspace(0.0, 4.0, nlat).astype(np.float32)
        lon = np.linspace(0.0, 4.0, nlon).astype(np.float32)
        t = np.array([0.0, 1e6], dtype=np.float32)
        u = np.full((nt, nlat, nlon), 0.05, dtype=np.float32)
        v = np.full((nt, nlat, nlon), 0.05, dtype=np.float32)
        # The 2×2 block around (lat,lon)=(2,2) is all land.
        m = np.zeros((nlat, nlon), dtype=bool)
        m[1:4, 1:4] = True
        ds = Dataset.from_arrays(
            {"u": u, "v": v}, t=t, lat=lat, lon=lon, masks={"u": m, "v": m},
        )

        def term(t, y, args):
            ds_ = args
            return jnp.array(
                [ds_["u"].interp(t, y[0], y[1]),
                 ds_["v"].interp(t, y[0], y[1])]
            )

        y0 = jnp.array([2.0, 2.0], dtype=jnp.float32)  # dead centre of the bay
        traj = solve(term, y0, jnp.array(0.0), 10, 10.0, 10.0, solver=Heun(), args=ds)
        # Final position equals start (zero velocity all along).
        assert float(traj[-1, 0]) == pytest.approx(2.0, abs=1e-5)
        assert float(traj[-1, 1]) == pytest.approx(2.0, abs=1e-5)
        # And no NaN anywhere.
        assert bool(jnp.all(jnp.isfinite(traj))) is True


class TestTransientNaNWarning:
    """NaN at some-but-not-all time steps looks like missing data, not land."""

    def _coords(self):
        t = np.linspace(0.0, 3600.0, 3)
        lat = np.linspace(0.0, 2.0, 3)
        lon = np.linspace(10.0, 13.0, 4)
        return t, lat, lon

    def test_transient_nan_warns(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 3, 4), dtype=np.float32)
        u[1, 0, 0] = np.nan  # NaN at one time step only: a gap, not land
        with pytest.warns(UserWarning, match="some but not all time steps"):
            ds = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
        # The cell is still masked (conservative) and zero-filled.
        assert bool(ds["u"].mask[0, 0]) is True
        assert not bool(jnp.isnan(ds["u"].values).any())

    def test_time_invariant_nan_does_not_warn(self):
        import warnings as _warnings

        t, lat, lon = self._coords()
        u = np.ones((3, 3, 4), dtype=np.float32)
        u[:, 0, 0] = np.nan  # land: NaN at every time step
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            ds = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
        assert bool(ds["u"].mask[0, 0]) is True

    def test_explicit_mask_silences_warning(self):
        import warnings as _warnings

        t, lat, lon = self._coords()
        u = np.ones((3, 3, 4), dtype=np.float32)
        u[1, 0, 0] = np.nan
        mask = np.zeros((3, 4), dtype=bool)
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            ds = Dataset.from_arrays(
                {"u": u}, t=t, lat=lat, lon=lon, masks={"u": mask}
            )
        assert ds["u"].mask is not None
        assert not bool(ds["u"].mask.any())
        
        
class TestUnknownMaskKeys:
    """A typo'd masks key must raise instead of silently falling back to NaN inference."""

    def test_from_arrays_unknown_key_raises(self):
        t = np.linspace(0.0, 3600.0, 2)
        lat = np.linspace(0.0, 2.0, 3)
        lon = np.linspace(10.0, 13.0, 4)
        u = np.ones((2, 3, 4), dtype=np.float32)
        mask = np.zeros((3, 4), dtype=bool)
        with pytest.raises(ValueError, match=r"masks contains keys \['U'\]"):
            Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon, masks={"U": mask})

    def test_from_arrays_cgrid_unknown_key_raises(self):
        t = np.linspace(0.0, 3600.0, 2)
        lat = np.linspace(0.0, 4.0, 5)
        lon = np.linspace(0.0, 5.0, 6)
        u = np.ones((2, 5, 5), dtype=np.float32)
        v = np.ones((2, 4, 6), dtype=np.float32)
        mask = np.zeros((5, 5), dtype=bool)
        with pytest.raises(ValueError, match=r"masks contains keys \['uo_typo'\]"):
            Dataset.from_arrays_cgrid(
                t, lat, lon,
                vectors={"current": {"u": ("uo", u), "v": ("vo", v)}},
                masks={"uo_typo": mask},
            )

    def test_from_arrays_known_key_still_works(self):
        t = np.linspace(0.0, 3600.0, 2)
        lat = np.linspace(0.0, 2.0, 3)
        lon = np.linspace(10.0, 13.0, 4)
        u = np.ones((2, 3, 4), dtype=np.float32)
        mask = np.zeros((3, 4), dtype=bool)
        ds = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon, masks={"u": mask})
        assert ds["u"].mask is not None
