"""Tests for forcing.py: Field and Dataset."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from pastax import Heun, solve
from pastax.forcing import Dataset, Field
from pastax.grid import Grid


def make_synthetic_ds():
    times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]")
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([10.0, 11.0, 12.0])
    u_data = np.ones((3, 3, 3), dtype=np.float32)
    v_data = np.zeros((3, 3, 3), dtype=np.float32)
    return xr.Dataset(
        {"u": (["time", "lat", "lon"], u_data),
         "v": (["time", "lat", "lon"], v_data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )


def make_field(n=5, values_fill=1.0):
    lats = jnp.linspace(0.0, 4.0, n)
    lons = jnp.linspace(10.0, 14.0, n)
    t_coords = jnp.linspace(0.0, 4 * 86400.0, n)
    values = jnp.full((n, n, n), values_fill)
    return Field.standalone(values=values, t_coords=t_coords, lat_coords=lats, lon_coords=lons)


class TestField:
    def test_interp_returns_scalar(self):
        field = make_field()
        v = field.interp(jnp.array(43200.0), jnp.array(11.0), jnp.array(1.0))
        assert v.shape == ()
        assert float(v) == pytest.approx(1.0)

    def test_neighborhood_shape_default(self):
        field = make_field(n=7)
        patch = field.neighborhood(
            jnp.array(3 * 86400.0), jnp.array(12.0), jnp.array(2.0)
        )
        assert patch.shape == (3, 3, 3)  # 2*1+1 in each dim

    def test_neighborhood_shape_custom(self):
        field = make_field(n=9)
        patch = field.neighborhood(
            jnp.array(0.0), jnp.array(10.0), jnp.array(0.0),
            t_window=2, lat_window=1, lon_window=3,
        )
        assert patch.shape == (5, 3, 7)

    def test_neighborhood_values_uniform_field(self):
        field = make_field(n=7, values_fill=3.14)
        patch = field.neighborhood(
            jnp.array(3 * 86400.0), jnp.array(12.0), jnp.array(2.0)
        )
        assert jnp.allclose(patch, jnp.full_like(patch, 3.14))

    def test_neighborhood_clamped_at_boundary(self):
        # Query at the very start — window should still have the right shape
        field = make_field(n=7)
        patch = field.neighborhood(
            jnp.array(0.0), jnp.array(10.0), jnp.array(0.0)
        )
        assert patch.shape == (3, 3, 3)


class TestFieldLonPeriodic:
    """Longitude wrap-around in Field.interp and Field.neighborhood."""

    def _periodic_field(self):
        # Global longitude grid: 4 cells of 90° spanning [0, 360)
        lats = jnp.array([0.0, 1.0])
        lons = jnp.array([0.0, 90.0, 180.0, 270.0])
        t_coords = jnp.array([0.0, 1.0])
        # values[t, lat, lon] encode the lon index directly
        slab = jnp.broadcast_to(jnp.array([0.0, 1.0, 2.0, 3.0]), (2, 4))
        values = jnp.stack([slab, slab])  # (n_t=2, n_lat=2, n_lon=4)
        return Field.standalone(
            values=values,
            t_coords=t_coords,
            lat_coords=lats,
            lon_coords=lons,
            lon_period=360.0,
        )

    def test_interp_wraps(self):
        field = self._periodic_field()
        v = field.interp(jnp.array(0.5), jnp.array(315.0), jnp.array(0.0))
        # midpoint between lon-index 3 and lon-index 0
        assert float(v) == pytest.approx(1.5)

    def test_neighborhood_wraps_at_zero(self):
        field = self._periodic_field()
        # Centre on lon=0° (index 0), window=1 → indices [3, 0, 1]
        patch = field.neighborhood(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
            t_window=0, lat_window=0, lon_window=1,
        )
        assert patch.shape == (1, 1, 3)
        assert jnp.allclose(patch[0, 0], jnp.array([3.0, 0.0, 1.0]))

    def test_neighborhood_wraps_at_high_end(self):
        field = self._periodic_field()
        # Centre on lon=270° (index 3), window=1 → indices [2, 3, 0]
        patch = field.neighborhood(
            jnp.array(0.0), jnp.array(270.0), jnp.array(0.0),
            t_window=0, lat_window=0, lon_window=1,
        )
        assert jnp.allclose(patch[0, 0], jnp.array([2.0, 3.0, 0.0]))

    def test_neighborhood_negative_lon_wraps(self):
        field = self._periodic_field()
        # -90° == 270° (index 3); window=1 → indices [2, 3, 0]
        patch = field.neighborhood(
            jnp.array(0.0), jnp.array(-90.0), jnp.array(0.0),
            t_window=0, lat_window=0, lon_window=1,
        )
        assert jnp.allclose(patch[0, 0], jnp.array([2.0, 3.0, 0.0]))

    def test_non_periodic_neighborhood_clamps_at_zero(self):
        # Without lon_period the existing clamp behaviour is preserved
        lats = jnp.array([0.0, 1.0])
        lons = jnp.array([0.0, 90.0, 180.0, 270.0])
        t_coords = jnp.array([0.0, 1.0])
        slab = jnp.broadcast_to(jnp.array([0.0, 1.0, 2.0, 3.0]), (2, 4))
        values = jnp.stack([slab, slab])
        field = Field.standalone(
            values=values,
            t_coords=t_coords,
            lat_coords=lats,
            lon_coords=lons,
        )
        patch = field.neighborhood(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
            t_window=0, lat_window=0, lon_window=1,
        )
        # clamp to start: indices [0, 1, 2]
        assert jnp.allclose(patch[0, 0], jnp.array([0.0, 1.0, 2.0]))

    def test_neighborhood_jit_compatible(self):
        field = self._periodic_field()

        @jax.jit
        def f(lon):
            return field.neighborhood(
                jnp.array(0.0), lon, jnp.array(0.0),
                t_window=0, lat_window=0, lon_window=1,
            )

        patch = f(jnp.array(0.0))
        assert jnp.allclose(patch[0, 0], jnp.array([3.0, 0.0, 1.0]))


class TestDataset:
    def test_from_xarray_loads_fields(self):
        ds = make_synthetic_ds()
        dataset = Dataset.from_xarray(
            ds,
            fields={"u": "u", "v": "v"},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        assert "u" in dataset.fields
        assert "v" in dataset.fields

    def test_from_xarray_interp_uniform_field(self):
        ds = make_synthetic_ds()
        dataset = Dataset.from_xarray(
            ds,
            fields={"u": "u"},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        field = dataset["u"]
        # datetime64 time is rebased to seconds since the first timestamp:
        # 2020-01-01T12:00:00 is 43200 s after the first stamp.
        v = field.interp(jnp.array(43200.0), jnp.array(11.0), jnp.array(1.0))
        assert float(v) == pytest.approx(1.0, abs=1e-5)

    def test_getitem(self):
        ds = make_synthetic_ds()
        dataset = Dataset.from_xarray(
            ds,
            fields={"u": "u"},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        assert isinstance(dataset["u"], Field)

    def test_neighborhood_returns_dict(self):
        ds = make_synthetic_ds()
        dataset = Dataset.from_xarray(
            ds,
            fields={"u": "u", "v": "v"},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        # Use a large-enough grid: ds has 3 points per axis, window=1 → need 3 points → OK
        patches = dataset.neighborhood(
            jnp.array(86400.0),  # 2020-01-02, rebased to the first timestamp
            jnp.array(11.0),
            jnp.array(1.0),
        )
        assert set(patches.keys()) == {"u", "v"}
        assert patches["u"].shape == (3, 3, 3)


class TestDatasetFromArrays:
    def _coords(self, n=4):
        t   = np.linspace(0.0, (n - 1) * 3600.0, n)
        lat = np.linspace(0.0, float(n - 1), n)
        lon = np.linspace(10.0, 10.0 + float(n - 1), n)
        return t, lat, lon

    def test_builds_dataset(self):
        t, lat, lon = self._coords()
        u = np.ones((4, 4, 4), dtype=np.float32)
        dataset = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
        assert "u" in dataset.fields
        assert isinstance(dataset["u"], Field)

    def test_interp_uniform_field(self):
        t, lat, lon = self._coords()
        u = np.ones((4, 4, 4), dtype=np.float32)
        dataset = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
        v = dataset["u"].interp(jnp.array(1800.0), jnp.array(11.5), jnp.array(1.5))
        assert float(v) == pytest.approx(1.0, abs=1e-5)

    def test_accepts_jax_arrays(self):
        t   = jnp.linspace(0.0, 3 * 3600.0, 4)
        lat = jnp.linspace(0.0, 3.0, 4)
        lon = jnp.linspace(10.0, 13.0, 4)
        u   = jnp.zeros((4, 4, 4))
        dataset = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
        assert dataset["u"].values.shape == (4, 4, 4)

    def test_lon_period_propagates_to_fields(self):
        t = np.linspace(0.0, 3600.0, 2)
        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 90.0, 180.0, 270.0])
        u = np.broadcast_to(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
                            (2, 2, 4))
        dataset = Dataset.from_arrays(
            {"u": u}, t=t, lat=lat, lon=lon, lon_period=360.0,
        )
        assert dataset["u"].lon_period == 360.0
        v = dataset["u"].interp(jnp.array(0.0), jnp.array(315.0), jnp.array(0.0))
        assert float(v) == pytest.approx(1.5)

    def test_from_arrays_accepts_datetime64(self):
        """datetime64 time is rebased to seconds since the Unix epoch."""
        t_dt = np.array(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
                        dtype="datetime64[D]")
        _, lat, lon = self._coords()
        u = np.ones((4, 4, 4), dtype=np.float32)
        dataset = Dataset.from_arrays({"u": u}, t=t_dt, lat=lat, lon=lon)
        epoch = t_dt.astype("datetime64[s]").astype(np.int64)
        assert jnp.allclose(
            dataset["u"].t_coords,
            jnp.asarray(epoch, dtype=dataset["u"].t_coords.dtype),
        )

    def test_from_arrays_datetime64_matches_from_xarray(self):
        """from_arrays with datetime64 and from_xarray must yield identical t_coords."""
        ds = make_synthetic_ds()
        ds_dataset = Dataset.from_xarray(
            ds,
            fields={"u": "u"},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        arr_dataset = Dataset.from_arrays(
            {"u": ds["u"].values},
            t=ds["time"].values,  # datetime64[D], passed in raw
            lat=ds["lat"].values,
            lon=ds["lon"].values,
        )
        assert jnp.allclose(ds_dataset["u"].t_coords, arr_dataset["u"].t_coords)

    def test_from_arrays_and_from_xarray_agree(self):
        """from_arrays and from_xarray must produce identical field values."""
        ds = make_synthetic_ds()
        ds_dataset = Dataset.from_xarray(
            ds,
            fields={"u": "u"},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        t = ds["time"].values.astype("datetime64[s]").astype(np.int64).astype(np.float32)
        arr_dataset = Dataset.from_arrays(
            {"u": ds["u"].values},
            t=t,
            lat=ds["lat"].values,
            lon=ds["lon"].values,
        )
        assert jnp.allclose(ds_dataset["u"].values, arr_dataset["u"].values)


def _uv_vectors(u, v, group="current", u_name="u", v_name="v"):
    """Wrap a single (u, v) pair into the new `vectors` dict shape."""
    return {group: {"u": (u_name, u), "v": (v_name, v)}}


class TestDatasetCGrid:
    """NEMO-convention Arakawa C-grid loaders and round-trip behaviour."""

    def _cgrid_inputs(self, nlat=6, nlon=8, nt=3):
        t   = np.linspace(0.0, (nt - 1) * 3600.0, nt)
        lat = np.linspace(-2.5, 2.5, nlat)
        lon = np.linspace(10.0, 17.0, nlon)
        u = np.ones((nt, nlat, nlon - 1), dtype=np.float32)
        v = np.zeros((nt, nlat - 1, nlon), dtype=np.float32)
        return t, lat, lon, u, v

    def test_builds_u_v_with_correct_stagger(self):
        t, lat, lon, u, v = self._cgrid_inputs()
        ds = Dataset.from_arrays_cgrid(t, lat, lon, _uv_vectors(u, v))
        assert ds["u"].stagger == "u_face"
        assert ds["v"].stagger == "v_face"
        assert ds["u"].values.shape == u.shape
        assert ds["v"].values.shape == v.shape

    def test_grid_metadata_is_c_grid(self):
        t, lat, lon, u, v = self._cgrid_inputs()
        ds = Dataset.from_arrays_cgrid(t, lat, lon, _uv_vectors(u, v))
        assert isinstance(ds.grid, Grid)
        assert ds.grid.stagger_type == "C"
        assert ds.grid.grid_type == "rectilinear"

    def test_auto_derived_coords_match_grid_helpers(self):
        t, lat, lon, u, v = self._cgrid_inputs()
        ds = Dataset.from_arrays_cgrid(t, lat, lon, _uv_vectors(u, v))
        u_lat_expected, u_lon_expected = ds.grid.u_face_coords()
        v_lat_expected, v_lon_expected = ds.grid.v_face_coords()
        assert jnp.allclose(ds["u"].lat_coords, u_lat_expected)
        assert jnp.allclose(ds["u"].lon_coords, u_lon_expected)
        assert jnp.allclose(ds["v"].lat_coords, v_lat_expected)
        assert jnp.allclose(ds["v"].lon_coords, v_lon_expected)

    def test_u_lon_shifted_half_cell_east(self):
        t, lat, lon, u, v = self._cgrid_inputs()
        ds = Dataset.from_arrays_cgrid(t, lat, lon, _uv_vectors(u, v))
        dlon = float(lon[1] - lon[0])
        assert jnp.allclose(ds["u"].lon_coords, jnp.asarray(lon[:-1] + 0.5 * dlon))

    def test_v_lat_shifted_half_cell_north(self):
        t, lat, lon, u, v = self._cgrid_inputs()
        ds = Dataset.from_arrays_cgrid(t, lat, lon, _uv_vectors(u, v))
        dlat = float(lat[1] - lat[0])
        assert jnp.allclose(ds["v"].lat_coords, jnp.asarray(lat[:-1] + 0.5 * dlat))

    def test_explicit_staggered_coords_override_autoderive(self):
        t, lat, lon, u, v = self._cgrid_inputs()
        custom_u_lon = jnp.asarray(lon[:-1] + 0.7)  # arbitrary, not half-cell
        ds = Dataset.from_arrays_cgrid(
            t, lat, lon, _uv_vectors(u, v), u_lon=custom_u_lon,
        )
        assert jnp.allclose(ds["u"].lon_coords, custom_u_lon)

    def test_tracers_attached_at_centre(self):
        t, lat, lon, u, v = self._cgrid_inputs()
        sst = np.full((3, 6, 8), 15.0, dtype=np.float32)
        ds = Dataset.from_arrays_cgrid(
            t, lat, lon, _uv_vectors(u, v), tracers={"sst": sst},
        )
        assert ds["sst"].stagger == "center"
        assert ds["sst"].values.shape == (3, 6, 8)
        assert jnp.allclose(ds["sst"].lat_coords, jnp.asarray(lat))
        assert jnp.allclose(ds["sst"].lon_coords, jnp.asarray(lon))

    def test_rejects_wrong_u_shape(self):
        t, lat, lon, u, v = self._cgrid_inputs()
        u_wrong = np.ones((3, 6, 8), dtype=np.float32)  # nlon instead of nlon-1
        with pytest.raises(ValueError, match=r"vectors\['current'\]\['u'\]"):
            Dataset.from_arrays_cgrid(t, lat, lon, _uv_vectors(u_wrong, v))

    def test_rejects_wrong_v_shape(self):
        t, lat, lon, u, v = self._cgrid_inputs()
        v_wrong = np.zeros((3, 6, 8), dtype=np.float32)  # nlat instead of nlat-1
        with pytest.raises(ValueError, match=r"vectors\['current'\]\['v'\]"):
            Dataset.from_arrays_cgrid(t, lat, lon, _uv_vectors(u, v_wrong))

    def test_rejects_wrong_tracer_shape(self):
        t, lat, lon, u, v = self._cgrid_inputs()
        bad = np.zeros((3, 5, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="tracers"):
            Dataset.from_arrays_cgrid(
                t, lat, lon, _uv_vectors(u, v), tracers={"sst": bad},
            )

    def test_rejects_empty_vectors(self):
        t, lat, lon, _, _ = self._cgrid_inputs()
        with pytest.raises(ValueError, match="at least one vector"):
            Dataset.from_arrays_cgrid(t, lat, lon, {})

    def test_rejects_duplicate_field_name_across_vectors(self):
        t, lat, lon, u, v = self._cgrid_inputs()
        # Same u-field name reused in a second vector.
        with pytest.raises(ValueError, match="Duplicate field name 'u'"):
            Dataset.from_arrays_cgrid(
                t, lat, lon,
                {
                    "current": {"u": ("u", u), "v": ("v", v)},
                    "wind":    {"u": ("u", u), "v": ("v_wind", v)},
                },
            )

    def test_rejects_tracer_name_colliding_with_vector(self):
        t, lat, lon, u, v = self._cgrid_inputs()
        sst = np.full((3, 6, 8), 15.0, dtype=np.float32)
        with pytest.raises(ValueError, match="Duplicate field name 'u'"):
            Dataset.from_arrays_cgrid(
                t, lat, lon, _uv_vectors(u, v), tracers={"u": sst},
            )

    def test_rejects_vectors_missing_uv_keys(self):
        t, lat, lon, u, v = self._cgrid_inputs()
        with pytest.raises(ValueError, match=r"vectors\['current'\] must declare"):
            Dataset.from_arrays_cgrid(
                t, lat, lon, {"current": {"u": ("u", u)}},
            )

    def test_multiple_vectors_share_grid(self):
        t, lat, lon, u, v = self._cgrid_inputs()
        # Surface current vs. 10-m wind on the same C-grid.
        u_wind = (2.0 * np.ones_like(u)).astype(np.float32)
        v_wind = (3.0 * np.ones_like(v)).astype(np.float32)
        ds = Dataset.from_arrays_cgrid(
            t, lat, lon,
            {
                "current": {"u": ("uo",  u),      "v": ("vo",  v)},
                "wind":    {"u": ("u10", u_wind), "v": ("v10", v_wind)},
            },
        )
        assert set(ds.fields) == {"uo", "vo", "u10", "v10"}
        for name in ("uo", "u10"):
            assert ds[name].stagger == "u_face"
            assert ds[name].values.shape == u.shape
        for name in ("vo", "v10"):
            assert ds[name].stagger == "v_face"
            assert ds[name].values.shape == v.shape
        # All U fields share staggered coords; same for V.
        assert jnp.allclose(ds["uo"].lon_coords, ds["u10"].lon_coords)
        assert jnp.allclose(ds["uo"].lat_coords, ds["u10"].lat_coords)
        assert jnp.allclose(ds["vo"].lon_coords, ds["v10"].lon_coords)
        assert jnp.allclose(ds["vo"].lat_coords, ds["v10"].lat_coords)
        # Distinct values per vector.
        assert float(ds["uo"].values.mean())  == pytest.approx(1.0)
        assert float(ds["u10"].values.mean()) == pytest.approx(2.0)
        assert float(ds["vo"].values.mean())  == pytest.approx(0.0)
        assert float(ds["v10"].values.mean()) == pytest.approx(3.0)

    def test_velocity_interp_picks_named_vector(self):
        t, lat, lon, u, v = self._cgrid_inputs()
        u_wind = (2.0 * np.ones_like(u)).astype(np.float32)
        v_wind = (3.0 * np.ones_like(v)).astype(np.float32)
        ds = Dataset.from_arrays_cgrid(
            t, lat, lon,
            {
                "current": {"u": ("uo",  u),      "v": ("vo",  v)},
                "wind":    {"u": ("u10", u_wind), "v": ("v10", v_wind)},
            },
        )
        q = (jnp.asarray(0.0), jnp.asarray(13.5), jnp.asarray(0.0))
        uv_current = ds.velocity_interp(*q, u_name="uo",  v_name="vo")
        uv_wind    = ds.velocity_interp(*q, u_name="u10", v_name="v10")
        assert float(uv_current[0]) == pytest.approx(1.0)  # U from "current"
        assert float(uv_current[1]) == pytest.approx(0.0)  # V from "current"
        assert float(uv_wind[0])    == pytest.approx(2.0)  # U from "wind"
        assert float(uv_wind[1])    == pytest.approx(3.0)  # V from "wind"

    def test_from_xarray_cgrid_matches_from_arrays_cgrid(self):
        t = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]")
        lat = np.linspace(-2.0, 2.0, 5)
        lon = np.linspace(0.0, 6.0, 7)
        u_data = np.full((3, 5, 6), 0.1, dtype=np.float32)
        v_data = np.full((3, 4, 7), 0.2, dtype=np.float32)
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
        from_xr = Dataset.from_xarray_cgrid(
            ds_xr,
            vectors={"current": {"u": ("u", "uo"), "v": ("v", "vo")}},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        from_arr = Dataset.from_arrays_cgrid(t, lat, lon, _uv_vectors(u_data, v_data))
        assert jnp.allclose(from_xr["u"].values, from_arr["u"].values)
        assert jnp.allclose(from_xr["v"].values, from_arr["v"].values)
        assert jnp.allclose(from_xr["u"].lon_coords, from_arr["u"].lon_coords)
        assert jnp.allclose(from_xr["v"].lat_coords, from_arr["v"].lat_coords)

    def test_from_xarray_cgrid_with_explicit_staggered_coords(self):
        t = np.linspace(0.0, 7200.0, 3)
        lat = np.linspace(0.0, 4.0, 5)
        lon = np.linspace(0.0, 6.0, 7)
        u_data = np.zeros((3, 5, 6), dtype=np.float32)
        v_data = np.zeros((3, 4, 7), dtype=np.float32)
        explicit_u_lon = lon[:-1] + 0.5
        ds_xr = xr.Dataset(
            {
                "uo": (["time", "lat", "lon_u"], u_data),
                "vo": (["time", "lat_v", "lon"], v_data),
            },
            coords={
                "time": t, "lat": lat, "lon": lon,
                "lon_u": explicit_u_lon,
                "lat_v": lat[:-1] + 0.5 * (lat[1] - lat[0]),
            },
        )
        ds = Dataset.from_xarray_cgrid(
            ds_xr,
            vectors={"current": {"u": ("u", "uo"), "v": ("v", "vo")}},
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
            staggered_coordinates={"u_lon": "lon_u"},
        )
        assert jnp.allclose(ds["u"].lon_coords, jnp.asarray(explicit_u_lon))

    def test_from_xarray_cgrid_multiple_vectors(self):
        t = np.linspace(0.0, 7200.0, 3)
        lat = np.linspace(0.0, 4.0, 5)
        lon = np.linspace(0.0, 6.0, 7)
        u_curr = np.full((3, 5, 6), 0.1, dtype=np.float32)
        v_curr = np.full((3, 4, 7), 0.2, dtype=np.float32)
        u_wind = np.full((3, 5, 6), 5.0, dtype=np.float32)
        v_wind = np.full((3, 4, 7), 7.0, dtype=np.float32)
        ds_xr = xr.Dataset(
            {
                "uo":  (["time", "lat", "lon_u"], u_curr),
                "vo":  (["time", "lat_v", "lon"], v_curr),
                "u10": (["time", "lat", "lon_u"], u_wind),
                "v10": (["time", "lat_v", "lon"], v_wind),
            },
            coords={
                "time": t, "lat": lat, "lon": lon,
                "lon_u": lon[:-1] + 0.5 * (lon[1] - lon[0]),
                "lat_v": lat[:-1] + 0.5 * (lat[1] - lat[0]),
            },
        )
        ds = Dataset.from_xarray_cgrid(
            ds_xr,
            vectors={
                "current": {"u": ("uo",  "uo"),  "v": ("vo",  "vo")},
                "wind":    {"u": ("u10", "u10"), "v": ("v10", "v10")},
            },
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
        )
        assert set(ds.fields) == {"uo", "vo", "u10", "v10"}
        assert float(ds["u10"].values.mean()) == pytest.approx(5.0)
        assert float(ds["v10"].values.mean()) == pytest.approx(7.0)

    def test_analytic_trajectory_linear_velocity(self):
        """U(lon)=α·lon, V(lat)=β·lat on a C-grid → trajectory grows
        exponentially: lon(t)=lon0·exp(αt), lat(t)=lat0·exp(βt).
        Bilinear-on-shifted-coords is exact for linear fields, so the
        only error left is the Heun integrator's O(h²)."""
        alpha = 1e-5
        beta  = 1e-5
        T = 5e4  # α·T = β·T = 0.5 → endpoint stays inside grid bounds
        nlat, nlon = 21, 21
        lat = np.linspace(0.5, 4.5, nlat).astype(np.float32)
        lon = np.linspace(0.5, 4.5, nlon).astype(np.float32)
        dlat = float(lat[1] - lat[0])
        dlon = float(lon[1] - lon[0])
        u_lon = lon[:-1] + 0.5 * dlon
        v_lat = lat[:-1] + 0.5 * dlat
        nt = 2
        t = np.linspace(0.0, 2 * T, nt).astype(np.float32)

        # U depends only on lon, V only on lat → broadcast across lat/time.
        u_arr = np.broadcast_to(
            alpha * u_lon[None, None, :], (nt, nlat, nlon - 1),
        ).astype(np.float32)
        v_arr = np.broadcast_to(
            beta * v_lat[None, :, None], (nt, nlat - 1, nlon),
        ).astype(np.float32)

        dataset = Dataset.from_arrays_cgrid(t, lat, lon, _uv_vectors(u_arr, v_arr))

        # Treat U/V as degrees/second directly so the test isolates the
        # C-grid interpolation + solver from the meters_to_degrees conversion.
        def term(t_, y, args):
            ds = args
            u = ds["u"].interp(t_, y[0], y[1])
            v = ds["v"].interp(t_, y[0], y[1])
            return jnp.array([u, v])  # [dlon/dt, dlat/dt]

        n_save = 500
        dt = T / n_save
        y0 = jnp.array([2.0, 2.0], dtype=jnp.float32)
        traj = solve(term, y0, jnp.array(0.0), n_save, dt, dt, solver=Heun(), args=dataset)

        lat_final = float(traj[-1, 1])
        lon_final = float(traj[-1, 0])
        assert lat_final == pytest.approx(2.0 * float(np.exp(beta  * T)), rel=1e-3)
        assert lon_final == pytest.approx(2.0 * float(np.exp(alpha * T)), rel=1e-3)


class TestPeriodicLonAscending:
    """lon_period demands ascending longitudes; descending ones must raise."""

    def test_descending_lon_with_period_raises(self):
        t = np.linspace(0.0, 3600.0, 2)
        lat = np.array([0.0, 1.0])
        lon = np.array([270.0, 180.0, 90.0, 0.0])  # descending
        u = np.ones((2, 2, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="strictly ascending longitude"):
            Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon, lon_period=360.0)

    def test_descending_lon_without_period_still_works(self):
        t = np.linspace(0.0, 3600.0, 2)
        lat = np.array([0.0, 1.0])
        lon = np.array([13.0, 12.0, 11.0, 10.0])  # descending, non-periodic
        u = np.broadcast_to(np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32), (2, 2, 4))
        ds = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
        # values encode lon - 10, so interp at lon=11.5 must give 1.5
        v = ds["u"].interp(jnp.array(0.0), jnp.array(11.5), jnp.array(0.5))
        assert float(v) == pytest.approx(1.5, abs=1e-6)

    def test_cgrid_descending_lon_with_period_raises(self):
        t = np.linspace(0.0, 3600.0, 2)
        lat = np.linspace(0.0, 4.0, 5)
        lon = np.array([300.0, 240.0, 180.0, 120.0, 60.0, 0.0])  # descending
        u = np.ones((2, 5, 5), dtype=np.float32)
        v = np.ones((2, 4, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="strictly ascending longitude"):
            Dataset.from_arrays_cgrid(
                t, lat, lon,
                vectors={"current": {"u": ("uo", u), "v": ("vo", v)}},
                lon_period=360.0,
            )


class TestPeriodicLonSpan:
    """lon_period demands the grid span exactly one period (nlon*dlon == period)."""

    def test_regional_grid_mislabelled_periodic_raises(self):
        t = np.linspace(0.0, 3600.0, 2)
        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 1.0, 2.0, 3.0])  # spans 4 deg, not 360
        u = np.ones((2, 2, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="span exactly one|span exactly"):
            Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon, lon_period=360.0)

    def test_duplicate_wrap_endpoint_raises(self):
        """Including the wrap endpoint (nlon+1 points, last == first + period)
        overshoots one period by a cell and must be rejected."""
        t = np.linspace(0.0, 3600.0, 2)
        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 90.0, 180.0, 270.0, 360.0])  # 5 pts incl. wrap
        u = np.ones((2, 2, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="span exactly one|duplicate wrap"):
            Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon, lon_period=360.0)

    def test_correct_span_accepted(self):
        t = np.linspace(0.0, 3600.0, 2)
        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 90.0, 180.0, 270.0])  # 4 * 90 == 360
        u = np.ones((2, 2, 4), dtype=np.float32)
        ds = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon, lon_period=360.0)
        assert ds["u"].lon_period == 360.0

    def test_cgrid_mislabelled_periodic_raises(self):
        t = np.linspace(0.0, 3600.0, 2)
        lat = np.linspace(0.0, 4.0, 5)
        lon = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])  # spans 6 deg, not 360
        u = np.ones((2, 5, 5), dtype=np.float32)
        v = np.ones((2, 4, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="span exactly one|span exactly"):
            Dataset.from_arrays_cgrid(
                t, lat, lon,
                vectors={"current": {"u": ("uo", u), "v": ("vo", v)}},
                lon_period=360.0,
            )


class TestLoadersAreTracerSafe:
    """The loaders must be callable from inside a traced (jit/vmap) region.

    Host-side validations that read concrete values (the ascending-lon check
    and NaN-mask inference) are skipped under tracing; the caller then owns
    coordinate correctness and passes explicit masks when needed.
    """

    _lat = np.array([0.0, 1.0, 2.0, 3.0])
    _lon = np.array([0.0, 90.0, 180.0, 270.0])
    _t = np.array([0.0, 3600.0])

    def test_from_arrays_jit_with_lon_period(self):
        u = np.ones((2, 4, 4), dtype=np.float32)

        def build_and_interp(u, lon, q):
            ds = Dataset.from_arrays(
                {"u": u}, t=self._t, lat=self._lat, lon=lon, lon_period=360.0
            )
            return ds["u"].interp(jnp.array(0.0), q, jnp.array(1.0))

        v = jax.jit(build_and_interp)(
            jnp.asarray(u), jnp.asarray(self._lon), jnp.array(315.0)
        )
        assert float(v) == pytest.approx(1.0, abs=1e-6)

    def test_from_arrays_jit_nan_zeroed_mask_deferred(self):
        u = np.ones((2, 4, 4), dtype=np.float32)
        u[:, 0, 0] = np.nan  # would infer a mask eagerly

        def total(u):
            ds = Dataset.from_arrays({"u": u}, t=self._t, lat=self._lat, lon=self._lon)
            return ds["u"].values.sum()

        # NaN is still replaced with 0 under trace (2 cells) → 32 - 2 = 30.
        assert float(jax.jit(total)(jnp.asarray(u))) == pytest.approx(30.0)

        # Eager on the same input still infers the mask (structure differs).
        eager = Dataset.from_arrays(
            {"u": u}, t=self._t, lat=self._lat, lon=self._lon
        )
        assert eager["u"].mask is not None

    def test_from_arrays_jit_with_explicit_mask(self):
        u = np.ones((2, 4, 4), dtype=np.float32)
        m = np.zeros((4, 4), dtype=bool)
        m[0, 0] = True

        def build(u, mask):
            ds = Dataset.from_arrays(
                {"u": u}, t=self._t, lat=self._lat, lon=self._lon, masks={"u": mask}
            )
            return ds["u"].mask

        out = jax.jit(build)(jnp.asarray(u), jnp.asarray(m))
        assert bool(out[0, 0]) is True

    def test_from_arrays_cgrid_jit(self):
        clat = np.linspace(0.0, 4.0, 5)
        clon = np.linspace(0.0, 5.0, 6)
        uc = np.ones((2, 5, 5), dtype=np.float32)
        vc = np.ones((2, 4, 6), dtype=np.float32)

        def build(uc, vc):
            ds = Dataset.from_arrays_cgrid(
                self._t, clat, clon,
                vectors={"current": {"u": ("u", uc), "v": ("v", vc)}},
            )
            return ds["u"].interp(jnp.array(0.0), jnp.array(2.5), jnp.array(2.0))

        v = jax.jit(build)(jnp.asarray(uc), jnp.asarray(vc))
        assert float(v) == pytest.approx(1.0, abs=1e-6)


class TestFromArraysShapeValidation:
    def _coords(self):
        t   = np.linspace(0.0, 3 * 3600.0, 4)
        lat = np.linspace(0.0, 2.0, 3)
        lon = np.linspace(10.0, 14.0, 5)
        return t, lat, lon

    def test_matching_shape_passes(self):
        t, lat, lon = self._coords()
        u = np.ones((4, 3, 5), dtype=np.float32)
        dataset = Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
        assert dataset["u"].values.shape == (4, 3, 5)

    def test_transposed_lat_lon_raises(self):
        t, lat, lon = self._coords()
        u = np.ones((4, 5, 3), dtype=np.float32)  # (time, lon, lat) by mistake
        with pytest.raises(ValueError, match=r"fields\['u'\].*expected shape"):
            Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)

    def test_wrong_time_length_raises(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 3, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="expected shape"):
            Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)

    def test_2d_field_raises(self):
        t, lat, lon = self._coords()
        u = np.ones((3, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="expected shape"):
            Dataset.from_arrays({"u": u}, t=t, lat=lat, lon=lon)
