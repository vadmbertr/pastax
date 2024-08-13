import numpy as np
import xarray as xr
import xgcm

from ._preprocessing import Preprocessing


class Field:
    id: str = ""

    def __init__(self, is_cgrid: bool = False, preprocessing: Preprocessing = None):
        self.dataset: xr.Dataset | None = None
        self.grid: xgcm.Grid | None = None
        self.is_cgrid: bool = is_cgrid | (preprocessing is not None)
        self.preprocessing: Preprocessing = preprocessing

        if self.preprocessing is not None:
            self.id += f"_{self.preprocessing.id}"

    def _do_load_data(
            self,
            min_time: int, max_time: int,
            min_latitude: float, max_latitude: float,
            min_longitude: float, max_longitude: float
    ) -> xr.Dataset:
        raise NotImplementedError

    def __load_data_callback(self, ds: xr.Dataset) -> xr.Dataset:
        if self.preprocessing is not None:
            ds = self.preprocessing(ds)

        return ds

    def load_data(
            self,
            min_time: int, max_time: int,
            min_latitude: float, max_latitude: float,
            min_longitude: float, max_longitude: float
    ):
        ds = self._do_load_data(min_time, max_time, min_latitude, max_latitude, min_longitude, max_longitude)
        ds = self.__load_data_callback(ds)

        self.dataset = ds

        if self.is_cgrid:
            self.grid = xgcm.Grid(
                self.dataset,
                coords={
                    "X": {"center": "lon_t", "right": "lon_u"},
                    "Y": {"center": "lat_t", "right": "lat_v"}
                }
            )

    def __interpolate(self, da: xr.DataArray, axis: str):
        da_interp = self.grid.interp(da, axis=axis, boundary="extend")
        da = da_interp.where(np.isfinite(da_interp), da.data)  # land boundary condition

        return da

    def get_ssh(self) -> xr.DataArray:
        return self.dataset.ssh

    def get_u(self, interpolate: bool = False) -> xr.DataArray:
        u = self.dataset.u
        if interpolate & self.is_cgrid:
            u = self.__interpolate(u, axis="X")

        return u

    def get_v(self, interpolate: bool = False) -> xr.DataArray:
        v = self.dataset.v
        if interpolate & self.is_cgrid:
            v = self.__interpolate(v, axis="Y")

        return v

    def get_uv(self) -> xr.DataArray:
        u = self.get_u(interpolate=True)
        v = self.get_v(interpolate=True)

        return (u ** 2 + v ** 2) ** (1 / 2)

    def get_sst(self) -> xr.DataArray:
        return self.dataset.sst
