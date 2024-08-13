from functools import partial
from typing import Callable, Dict, Literal

import clouddrift as cd
import numpy as np
import xarray as xr

from .._preprocessor import Preprocessor


class GDP(Preprocessor):
    def __init__(self, **kwargs):
        self.func: Callable[[xr.Dataset], xr.Dataset] = partial(self._func, **kwargs)

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        ds = self.func(ds)
        print(f"# traj: {ds.traj.size} ; # obs: {ds.obs.size}")
        return ds

    @staticmethod
    def _func(ds: xr.Dataset, **kwargs) -> xr.Dataset:
        raise NotImplementedError


class DrifterIdx(GDP):
    @staticmethod
    def _func(ds: xr.Dataset, indexes: [int, ...] = None) -> xr.Dataset:
        print(
            f"Subsetting on drifter indexes...\n"
            f"(indexes: {indexes})"
        )

        if indexes is None:
            return ds

        ds.id.load()
        return cd.ragged.subset(ds, {"id": indexes}, row_dim_name="traj")


class DeployBbox(GDP):
    @staticmethod
    def _func(
            ds: xr.Dataset,
            min_lat: float = None,
            max_lat: float = None,
            min_lon: float = None,
            max_lon: float = None
    ) -> xr.Dataset:
        print(  # noqa
            f"Subsetting on drifter deploy location...\n"
            f"(min_lat, max_lat, min_lon, max_lon: {min_lat}, {max_lat}, {min_lon}, {max_lon})"
        )

        def compute_mask(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
            return (min_lat <= lat) & (lat <= max_lat) & (min_lon <= lon) & (lon <= max_lon)

        if (min_lon is None) | (max_lon is None) | (min_lat is None) | (max_lat is None):
            return ds

        ds.deploy_lat.load()
        ds.deploy_lon.load()
        return cd.ragged.subset(ds, {("deploy_lat", "deploy_lon"): compute_mask}, row_dim_name="traj")


class GPSLocationType(GDP):
    @staticmethod
    def _func(ds: xr.Dataset, **kwargs) -> xr.Dataset:
        print("Subsetting to GPS location type...")

        ds.location_type.load()
        return cd.ragged.subset(ds, {"location_type": True}, row_dim_name="traj")  # True means GPS / False Argos


class SVPBuoyTypes(GDP):
    @staticmethod
    def _func(ds: xr.Dataset, **kwargs) -> xr.Dataset:
        print("Subsetting to SVP buoy types...")

        ds.typebuoy.load()
        return cd.ragged.subset(
            ds,
            {"typebuoy": lambda tb: np.char.find(tb.astype(str), "SVP") != -1},
            row_dim_name="traj"
        )


class Time(GDP):
    @staticmethod
    def _func(ds: xr.Dataset, from_datetime: str = "2000", to_datetime: str = "2023-06-07") -> xr.Dataset:
        print(f"Subsetting to post {from_datetime} deployments...")

        ds.deploy_date.load()
        ds = cd.ragged.subset(
            ds,
            {"deploy_date": lambda dt: dt >= np.datetime64(from_datetime)},
            row_dim_name="traj"
        )

        print(f"Subsetting to pre {to_datetime} observations...")

        ds.time.load()
        ds = cd.ragged.subset(ds, {"time": lambda dt: dt < np.datetime64(to_datetime)}, row_dim_name="traj")
        return ds


class LocationBbox(GDP):
    @staticmethod
    def _func(
            ds: xr.Dataset,
            min_lat: float = None,
            max_lat: float = None,
            min_lon: float = None,
            max_lon: float = None
    ) -> xr.Dataset:
        print(  # noqa
            f"Subsetting on observation location...\n"
            f"(min_lat, max_lat, min_lon, max_lon: {min_lat}, {max_lat}, {min_lon}, {max_lon})"
        )

        def compute_mask(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
            return (min_lat <= lat) & (lat <= max_lat) & (min_lon <= lon) & (lon <= max_lon)

        if (min_lon is None) | (max_lon is None) | (min_lat is None) | (max_lat is None):
            return ds

        ds.lat.load()
        ds.lon.load()
        return cd.ragged.subset(ds, {("lat", "lon"): compute_mask}, row_dim_name="traj")


class Drogued(GDP):
    @staticmethod
    def _func(ds: xr.Dataset, **kwargs) -> xr.Dataset:
        print("Subsetting to drogued observations...")

        ds.drogue_status.load()
        return cd.ragged.subset(ds, {"drogue_status": True}, row_dim_name="traj")


class FiniteValue(GDP):
    @staticmethod
    def _func(ds: xr.Dataset, **kwargs) -> xr.Dataset:
        print("Subsetting to finite value observations...")

        ds.lat.load()
        ds = cd.ragged.subset(ds, {"lat": np.isfinite}, row_dim_name="traj")
        ds.lon.load()
        ds = cd.ragged.subset(ds, {"lon": np.isfinite}, row_dim_name="traj")
        ds.vn.load()
        ds = cd.ragged.subset(ds, {"vn": np.isfinite}, row_dim_name="traj")
        ds.ve.load()
        ds = cd.ragged.subset(ds, {"ve": np.isfinite}, row_dim_name="traj")
        ds.time.load()
        ds = cd.ragged.subset(ds, {"time": lambda arr: ~np.isnat(arr)}, row_dim_name="traj")
        return ds


class Outlier(GDP):
    @staticmethod
    def _func(ds: xr.Dataset, velocity_cutoff: float = 10, latlon_err_cutoff: float = .5) -> xr.Dataset:
        print(
            f"Subsetting to plausible value observations...\n"
            f"(velocity_cutoff: {velocity_cutoff}m/s, latlon_err_cutoff: {latlon_err_cutoff}°)"
        )

        def _velocity_cutoff(arr: xr.DataArray) -> xr.DataArray:
            return np.abs(arr) <= velocity_cutoff  # m/s

        def _err_cutoff(arr: xr.DataArray) -> xr.DataArray:
            return arr <= latlon_err_cutoff  # °

        ds.vn.load()
        ds = cd.ragged.subset(ds, {"vn": _velocity_cutoff}, row_dim_name="traj")
        ds.ve.load()
        ds = cd.ragged.subset(ds, {"ve": _velocity_cutoff}, row_dim_name="traj")
        ds.err_lat.load()
        ds = cd.ragged.subset(ds, {"err_lat": _err_cutoff}, row_dim_name="traj")
        ds.err_lon.load()
        ds = cd.ragged.subset(ds, {"err_lon": _err_cutoff}, row_dim_name="traj")
        return ds


class Chunk(GDP):
    @staticmethod
    def _func(ds: xr.Dataset, n_days: int = 9, dt: np.timedelta64 = None) -> xr.Dataset:
        print(
            f"Chunking in equally sampled trajectories...\n"
            f"(n_days: {n_days}, dt: {dt})"
        )

        def ragged_chunk(_arr: xr.DataArray | np.ndarray, _row_size: np.ndarray[int], _chunk_size: int) -> np.ndarray:
            return cd.ragged.apply_ragged(cd.ragged.chunk, _arr, _row_size, _chunk_size)  # noqa

        if dt is None:
            dt = (ds.isel(traj=0).time[1] - ds.isel(traj=0).time[0])

        row_size = cd.ragged.segment(ds.time, dt, ds.rowsize)  # if holes, divide into segments
        chunk_size = int(n_days / (dt / np.timedelta64(1, "D")))

        # chunk along `obs` dimension (data)
        data = dict([(d, ragged_chunk(ds[d], row_size, chunk_size).flatten())
                     for d in ["time", "lat", "lon", "err_lat", "err_lon"]])

        # chunk along `traj` dimension (metadata)
        metadata = dict([(md, ragged_chunk(np.repeat(ds[md], ds.rowsize), row_size, chunk_size)[:, 0].flatten())
                         for md in ["id", "deploy_date", "deploy_lat", "deploy_lon", "typebuoy"]])
        metadata["rowsize"] = np.full(metadata["id"].size, chunk_size)  # noqa - after chunking the rowsize is constant

        # create RaggedArray
        attrs_global = ds.attrs
        name_dims: Dict[str, Literal["rows", "obs"]] = {"traj": "rows", "obs": "obs"}
        coords = {"id": np.arange(metadata["id"].size), "time": data.pop("time")}
        coord_dims = {}
        attrs_variables = {}

        for var in ds.coords.keys():
            var = str(var)
            coord_dims[var] = str(ds[var].dims[-1])
            attrs_variables[var] = ds[var].attrs

        for var in data.keys():
            attrs_variables[var] = ds[var].attrs

        for var in metadata.keys():
            attrs_variables[var] = ds[var].attrs

        metadata["drifter_id"] = metadata["id"]  # noqa
        del metadata["id"]
        attrs_variables["drifter_id"] = attrs_variables["id"]
        del attrs_variables["id"]

        ragged_array = cd.RaggedArray(coords, metadata, data, attrs_global, attrs_variables, name_dims, coord_dims)

        # convert it back to xarray
        ds = ragged_array.to_xarray()

        return ds
