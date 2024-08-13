import os

import cachier
import copernicusmarine as cm
from dotenv import load_dotenv
import numpy as np
import xarray as xr

from .._field import Field
from .._preprocessing import Preprocessing


class Duacs(Field):
    id: str = "duacs"

    def __init__(
            self,
            username_env_var: str = "COPERNICUS_MARINE_SERVICE_USERNAME",
            password_env_var: str = "COPERNICUS_MARINE_SERVICE_PASSWORD",
            dataset_id: str = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D",
            dataset_version="202112",
            dataset_part="default",
            service="arco-time-series",
            disable_caching: bool = False,
            is_cgrid: bool = False,
            preprocessing: Preprocessing = None
    ):
        super().__init__(is_cgrid=is_cgrid | (preprocessing is not None), preprocessing=preprocessing)

        self.full_dataset = self.__open_cmes_dataset(
            username_env_var, password_env_var, dataset_id, dataset_version, dataset_part, service, disable_caching
        )

    @staticmethod
    def __open_cmes_dataset(
            username_env_var: str = "COPERNICUS_MARINE_SERVICE_USERNAME",
            password_env_var: str = "COPERNICUS_MARINE_SERVICE_PASSWORD",
            dataset_id: str = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D",
            dataset_version="202112",
            dataset_part="default",
            service="arco-time-series",
            disable_caching: bool = True
    ) -> xr.Dataset:
        print("Loading Aviso/DUACS data from CMEMS API...")

        if disable_caching:
            cachier.disable_caching()
        load_dotenv(override=True)
        cm.login(os.environ[username_env_var], os.environ[password_env_var], overwrite_configuration_file=True)

        ds = cm.open_dataset(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            dataset_part=dataset_part,
            service=service,
            variables=["adt", "ugos", "vgos"]
        )

        ds = ds.rename({
            "adt": "ssh",
            "ugos": "u",
            "vgos": "v",
            "longitude": "lon_t",
            "latitude": "lat_t"
        })

        return ds

    def _do_load_data(
            self,
            min_time: int, max_time: int,
            min_latitude: float, max_latitude: float,
            min_longitude: float, max_longitude: float
    ) -> xr.Dataset:
        print("Subsetting Aviso/DUACS data...")

        min_longitude = max(-179.875, min_longitude)
        max_longitude = min(179.875, max_longitude)
        min_latitude = max(-89.875, min_latitude)
        max_latitude = min(89.875, max_latitude)

        ds = self.full_dataset.sel(
            lon_t=slice(min_longitude, max_longitude),
            lat_t=slice(min_latitude, max_latitude),
            time=slice(np.datetime64(min_time, "s"), np.datetime64(max_time, "s"))
        )

        return ds
