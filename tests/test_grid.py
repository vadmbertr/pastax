"""Tests for the Grid metadata object and staggered-coordinate helpers."""

import jax.numpy as jnp
import pytest

from pastax import Dataset, Field, Grid


def _centre_grid():
    t = jnp.asarray([0.0, 3600.0, 7200.0])
    lat = jnp.linspace(-5.0, 5.0, 11)
    lon = jnp.linspace(0.0, 9.0, 10)
    return t, lat, lon


def test_grid_defaults_are_a_grid_rectilinear():
    t, lat, lon = _centre_grid()
    g = Grid(t_coords=t, lat_coords=lat, lon_coords=lon)
    assert g.grid_type == "rectilinear"
    assert g.stagger_type == "A"
    assert g.lon_period is None


def test_u_face_coords_shift_half_cell_in_lon():
    t, lat, lon = _centre_grid()
    g = Grid(t_coords=t, lat_coords=lat, lon_coords=lon, stagger_type="C")
    lat_u, lon_u = g.u_face_coords()
    assert lat_u.shape == lat.shape
    assert jnp.allclose(lat_u, lat)
    assert lon_u.shape == (lon.shape[0] - 1,)
    dlon = float(lon[1] - lon[0])
    assert jnp.allclose(lon_u, lon[:-1] + 0.5 * dlon)


def test_v_face_coords_shift_half_cell_in_lat():
    t, lat, lon = _centre_grid()
    g = Grid(t_coords=t, lat_coords=lat, lon_coords=lon, stagger_type="C")
    lat_v, lon_v = g.v_face_coords()
    assert lon_v.shape == lon.shape
    assert jnp.allclose(lon_v, lon)
    assert lat_v.shape == (lat.shape[0] - 1,)
    dlat = float(lat[1] - lat[0])
    assert jnp.allclose(lat_v, lat[:-1] + 0.5 * dlat)


def test_u_face_coords_periodic_includes_seam_face():
    """On a periodic grid every centre cell has an east face (the seam face
    included), so U is nlon-wide with faces a half-cell east of each centre."""
    t = jnp.asarray([0.0, 3600.0])
    lat = jnp.asarray([0.0, 1.0])
    lon = jnp.asarray([0.0, 90.0, 180.0, 270.0])  # nlon=4, period 360
    g = Grid(
        t_coords=t, lat_coords=lat, lon_coords=lon,
        stagger_type="C", lon_period=360.0,
    )
    lat_u, lon_u = g.u_face_coords()
    assert jnp.allclose(lat_u, lat)
    assert lon_u.shape == (lon.shape[0],)  # nlon, not nlon - 1
    assert jnp.allclose(lon_u, jnp.asarray([45.0, 135.0, 225.0, 315.0]))
    # spans exactly one period and stays equally spaced
    assert jnp.allclose(jnp.diff(lon_u), 90.0)


def test_u_face_coords_open_uses_interior_midpoints():
    """An open (seam-crossing) grid has no wrap and no seam face: U lives on the
    nlon-1 interior midpoints, like a bounded grid, computed on the unwrapped
    (ascending) centres."""
    t = jnp.asarray([0.0, 3600.0])
    lat = jnp.asarray([0.0, 1.0])
    lon = jnp.asarray([170.0, 175.0, 180.0, 185.0, 190.0])  # unwrapped seam grid
    g = Grid(
        t_coords=t, lat_coords=lat, lon_coords=lon,
        stagger_type="C", lon_period=360.0, lon_closed=False,
    )
    lat_u, lon_u = g.u_face_coords()
    assert jnp.allclose(lat_u, lat)
    assert lon_u.shape == (lon.shape[0] - 1,)  # nlon - 1, no seam face
    assert jnp.allclose(lon_u, jnp.asarray([172.5, 177.5, 182.5, 187.5]))


def test_closed_for_matches_lon_closed_for_all_staggers():
    t = jnp.asarray([0.0, 3600.0])
    lat = jnp.asarray([0.0, 1.0])
    lon = jnp.asarray([170.0, 175.0, 180.0, 185.0, 190.0])
    g_open = Grid(
        t_coords=t, lat_coords=lat, lon_coords=lon,
        stagger_type="C", lon_period=360.0, lon_closed=False,
    )
    for stagger in ("center", "u_face", "v_face"):
        assert g_open.closed_for(stagger) is False
    g_closed = Grid(
        t_coords=t, lat_coords=lat, lon_coords=jnp.asarray([0.0, 90.0, 180.0, 270.0]),
        stagger_type="C", lon_period=360.0,
    )
    for stagger in ("center", "u_face", "v_face"):
        assert g_closed.closed_for(stagger) is True


def test_period_for_returns_period_for_all_staggers_when_periodic():
    t = jnp.asarray([0.0, 3600.0])
    lat = jnp.asarray([0.0, 1.0])
    lon = jnp.asarray([0.0, 90.0, 180.0, 270.0])
    g = Grid(
        t_coords=t, lat_coords=lat, lon_coords=lon,
        stagger_type="C", lon_period=360.0,
    )
    assert g.period_for("center") == 360.0
    assert g.period_for("u_face") == 360.0
    assert g.period_for("v_face") == 360.0


def test_period_for_returns_none_for_all_staggers_when_bounded():
    t, lat, lon = _centre_grid()
    g = Grid(t_coords=t, lat_coords=lat, lon_coords=lon, stagger_type="C")
    assert g.period_for("center") is None
    assert g.period_for("u_face") is None
    assert g.period_for("v_face") is None


def test_staggered_coords_preserve_equal_spacing():
    """The C-grid first-order interp relies on shifted coords being equally
    spaced. Verify that explicitly so a future regression (e.g. switch to a
    non-uniform centre grid) is caught here."""
    t, lat, lon = _centre_grid()
    g = Grid(t_coords=t, lat_coords=lat, lon_coords=lon, stagger_type="C")
    _, lon_u = g.u_face_coords()
    lat_v, _ = g.v_face_coords()
    assert jnp.allclose(jnp.diff(lon_u), lon_u[1] - lon_u[0])
    assert jnp.allclose(jnp.diff(lat_v), lat_v[1] - lat_v[0])


def test_curvilinear_face_coords_raise_not_implemented():
    t, lat, lon = _centre_grid()
    lat_2d, lon_2d = jnp.meshgrid(lat, lon, indexing="ij")
    g = Grid(
        t_coords=t,
        lat_coords=lat_2d,
        lon_coords=lon_2d,
        grid_type="curvilinear",
        stagger_type="C",
    )
    with pytest.raises(NotImplementedError):
        g.u_face_coords()
    with pytest.raises(NotImplementedError):
        g.v_face_coords()


def test_field_stagger_defaults_to_center():
    t, lat, lon = _centre_grid()
    f = Field.standalone(
        values=jnp.zeros((3, 11, 10)),
        t_coords=t,
        lat_coords=lat,
        lon_coords=lon,
    )
    assert f.stagger == "center"


def test_field_can_be_tagged_u_face_or_v_face():
    t, lat, lon = _centre_grid()
    u = Field.standalone(
        values=jnp.zeros((3, 11, 9)),
        t_coords=t,
        lat_coords=lat,
        lon_coords=lon[:-1] + 0.5,
        stagger="u_face",
    )
    v = Field.standalone(
        values=jnp.zeros((3, 10, 10)),
        t_coords=t,
        lat_coords=lat[:-1] + 0.5,
        lon_coords=lon,
        stagger="v_face",
    )
    assert u.stagger == "u_face"
    assert v.stagger == "v_face"


def test_dataset_from_arrays_builds_a_grid():
    """A-grid datasets now carry a populated ``stagger_type="A"`` Grid, and
    every field reads its coordinates from that shared grid."""
    t, lat, lon = _centre_grid()
    ds = Dataset.from_arrays(
        {"u": jnp.zeros((3, 11, 10))}, t=t, lat=lat, lon=lon
    )
    assert isinstance(ds.grid, Grid)
    assert ds.grid.stagger_type == "A"
    assert ds.grid.grid_type == "rectilinear"
    # The field references the dataset's grid — no per-field coordinate copy.
    assert ds["u"].grid is ds.grid
    assert ds["u"].lat_coords is ds.grid.lat_coords


def test_dataset_accepts_grid_metadata():
    t, lat, lon = _centre_grid()
    g = Grid(t_coords=t, lat_coords=lat, lon_coords=lon, stagger_type="C")
    f = Field.standalone(
        values=jnp.zeros((3, 11, 10)),
        t_coords=t,
        lat_coords=lat,
        lon_coords=lon,
    )
    ds = Dataset(fields={"sst": f}, grid=g)
    assert ds.grid is g
    assert ds.grid.stagger_type == "C"


def test_field_interp_unchanged_for_center_stagger():
    """Adding the ``stagger`` field must not alter the interp result for
    existing A-grid Fields."""
    t, lat, lon = _centre_grid()
    values = jnp.arange(3 * 11 * 10, dtype=jnp.float32).reshape((3, 11, 10))
    f = Field.standalone(values=values, t_coords=t, lat_coords=lat, lon_coords=lon)
    v = f.interp(jnp.asarray(1800.0), jnp.asarray(4.5), jnp.asarray(0.0))
    assert jnp.isfinite(v)


def test_cgrid_fields_share_one_set_of_coords():
    """Grid is the single owner of coordinates: U/V fields read the grid's
    stored staggered arrays by identity, not per-field copies."""
    t = jnp.asarray([0.0, 3600.0, 7200.0])
    lat = jnp.linspace(-2.5, 2.5, 6)
    lon = jnp.linspace(10.0, 17.0, 8)
    u = jnp.ones((3, 6, 7))
    v = jnp.zeros((3, 5, 8))
    ds = Dataset.from_arrays_cgrid(
        t, lat, lon, {"current": {"u": ("u", u), "v": ("v", v)}},
    )
    # Every field points at the dataset's grid.
    assert ds["u"].grid is ds.grid
    assert ds["v"].grid is ds.grid
    # Face coords resolve to the grid's stored staggered arrays (no copy).
    assert ds["u"].lon_coords is ds.grid.u_lon_coords
    assert ds["u"].lat_coords is ds.grid.u_lat_coords
    assert ds["v"].lat_coords is ds.grid.v_lat_coords
    assert ds["v"].lon_coords is ds.grid.v_lon_coords
    # Time is shared by all roles.
    assert ds["u"].t_coords is ds.grid.t_coords
    assert ds["v"].t_coords is ds.grid.t_coords


def _periodic_cgrid():
    """Global periodic C-grid: nlon=4 centres of 90 deg; U values encode the
    face index so wrapping is observable."""
    t = jnp.asarray([0.0, 3600.0])
    clat = jnp.asarray([0.0, 1.0, 2.0])
    clon = jnp.asarray([0.0, 90.0, 180.0, 270.0])
    nt, nlat, nlon = 2, 3, 4
    u = jnp.broadcast_to(jnp.asarray([10.0, 11.0, 12.0, 13.0]), (nt, nlat, nlon))
    v = jnp.zeros((nt, nlat - 1, nlon))
    ds = Dataset.from_arrays_cgrid(
        t, clat, clon, {"cur": {"u": ("u", u), "v": ("v", v)}}, lon_period=360.0,
    )
    return ds


def test_periodic_cgrid_loader_accepts_nlon_wide_u():
    """A periodic C-grid takes nlon-wide U (one east face per centre cell) and
    stores the seam-inclusive face longitudes on the grid."""
    ds = _periodic_cgrid()
    assert ds["u"].values.shape == (2, 3, 4)          # nlon, not nlon - 1
    assert ds["u"].lon_coords.shape == (4,)
    assert jnp.allclose(ds["u"].lon_coords, jnp.asarray([45.0, 135.0, 225.0, 315.0]))
    assert ds["u"].lon_period == 360.0
    assert ds["v"].lon_period == 360.0


def test_periodic_cgrid_rejects_bounded_u_shape():
    """With lon_period set, an nlon-1-wide U array (the bounded shape) must be
    rejected — the seam face is now required."""
    t = jnp.asarray([0.0, 3600.0])
    clat = jnp.asarray([0.0, 1.0, 2.0])
    clon = jnp.asarray([0.0, 90.0, 180.0, 270.0])
    u_bounded = jnp.ones((2, 3, 3))   # nlon - 1
    v = jnp.zeros((2, 2, 4))
    with pytest.raises(ValueError, match="shape mismatch"):
        Dataset.from_arrays_cgrid(
            t, clat, clon, {"cur": {"u": ("u", u_bounded), "v": ("v", v)}},
            lon_period=360.0,
        )


def test_periodic_u_face_interp_wraps_across_seam():
    """U interpolation at lon=0 must blend the seam face (315deg -> 13) and the
    first face (45deg -> 10) to their midpoint, rather than extrapolate."""
    ds = _periodic_cgrid()
    v = ds["u"].interp(jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1.0))
    assert float(v) == pytest.approx(11.5, abs=1e-4)


def test_periodic_u_face_neighborhood_wraps_across_seam():
    ds = _periodic_cgrid()
    nb = ds["u"].neighborhood(
        jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1.0),
        t_window=0, lat_window=0, lon_window=1,
    )
    # centred on face index 0 (45deg): wrapped window is [seam=13, 10, 11]
    assert jnp.allclose(nb[0, 0], jnp.asarray([13.0, 10.0, 11.0]))


def test_standalone_u_face_periodic_wraps():
    """Field.standalone now stores lon_period for face staggers and wraps."""
    lon_u = jnp.asarray([45.0, 135.0, 225.0, 315.0])  # seam-inclusive faces
    lat = jnp.asarray([0.0, 1.0])
    t = jnp.asarray([0.0, 1.0])
    vals = jnp.broadcast_to(jnp.asarray([10.0, 11.0, 12.0, 13.0]), (2, 2, 4))
    f = Field.standalone(
        values=vals, t_coords=t, lat_coords=lat, lon_coords=lon_u,
        lon_period=360.0, stagger="u_face",
    )
    assert f.lon_period == 360.0
    v = f.interp(jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0))
    assert float(v) == pytest.approx(11.5, abs=1e-4)


def test_grid_backed_interp_is_jit_vmap_grad_safe():
    """The grid-backed interp path must trace, batch and differentiate
    cleanly (the Field reads coords from the shared Grid)."""
    import jax

    t, lat, lon = _centre_grid()
    values = jnp.arange(3 * 11 * 10, dtype=jnp.float32).reshape((3, 11, 10))
    ds = Dataset.from_arrays({"u": values}, t=t, lat=lat, lon=lon)

    @jax.jit
    def sample(lat_q, lon_q):
        return ds["u"].interp(jnp.asarray(1800.0), lon_q, lat_q)

    out = sample(jnp.asarray(0.0), jnp.asarray(4.5))
    assert jnp.isfinite(out)

    batched = jax.vmap(sample)(
        jnp.linspace(-4.0, 4.0, 5), jnp.linspace(1.0, 8.0, 5)
    )
    assert batched.shape == (5,)
    assert bool(jnp.all(jnp.isfinite(batched)))

    g = jax.grad(lambda p: sample(p[0], p[1]))(
        jnp.asarray([0.0, 4.5], dtype=jnp.float32)
    )
    assert bool(jnp.all(jnp.isfinite(g)))
