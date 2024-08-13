import io

import clouddrift as cd
import jax.numpy as jnp
import numpy as np
import xarray as xr

from ...io import ZarrStore
from ...trajectory import Trajectory
from ...utils.unit import time_in_seconds  # noqa
from .._drifter import Drifter
from .._preprocessing import Preprocessing


class GDP(Drifter):
    @staticmethod
    def _apply_preprocessing(ds: xr.Dataset, preprocessing: Preprocessing) -> xr.Dataset:
        print(f"# traj: {ds.traj.size} ; # obs: {ds.obs.size}")

        ds = Drifter._apply_preprocessing(ds, preprocessing)

        return ds

    def get_trajectory(self, drifter_idx: int = None) -> Trajectory:
        if drifter_idx is None:
            drifter_idx = np.random.randint(0, self.dataset.traj.size)

        ds = cd.ragged.subset(self.dataset, {"traj": drifter_idx}, row_dim_name="traj")

        drifter_id = drifter_idx
        if "drifter_id" in ds:
            drifter_id = int(ds.drifter_id.data)

        trajectory = Trajectory(
            jnp.stack((ds.lat.data, ds.lon.data)).T,
            time_in_seconds(ds.time.data),
            jnp.asarray(drifter_id, dtype=jnp.int32)
        )

        return trajectory


class GDP1h(GDP):
    id = "gdp1h"

    def _do_load_data(self) -> xr.Dataset:
        return cd.datasets.gdp1h()


class GDP6h(GDP):
    id = "gdp6h"

    def _do_load_data(self) -> xr.Dataset:
        return cd.datasets.gdp6h()


class GDP6hLocal(GDP6h):
    def __init__(
            self,
            preprocessing: Preprocessing = None,
            download_url: str = "https://www.aoml.noaa.gov/ftp/pub/phod/buoydata/gdp6h_ragged_may23.nc",
            data_path: str = "data",
            zarr_dir: str = "gdp6h.zarr"
    ):
        self.download_url = download_url
        self.store = ZarrStore(data_path, zarr_dir)
        super().__init__(preprocessing)

    def _do_load_data(self) -> xr.Dataset:
        def sanitize_date(arr: xr.DataArray) -> xr.DataArray:
            nat_index = arr < 0
            arr[nat_index.compute()] = np.nan
            return arr

        # the GDP6H inputs from clouddrift can not be opened "remotely", so we store it (for now)
        # (see https://github.com/Cloud-Drift/clouddrift/issues/363)
        if not self.store.exists():  # download dataset: takes time
            print("Downloading GDP6H data to local store...")

            # reused from cd.datasets.gdp6h
            buffer = io.BytesIO()
            cd.adapters.utils.download_with_progress([(f"{self.download_url}#mode=bytes", buffer, None)])
            reader = io.BufferedReader(buffer)  # noqa
            ds = xr.open_dataset(reader)
            ds = ds.rename_vars({"ID": "id"}).assign_coords({"id": ds.ID}).drop_vars(["ids"])
            ds.to_zarr(self.store.store, mode="w")

        print("Loading GDP6H data from local store...")

        ds = xr.open_zarr(self.store.store, decode_times=False)
        ds["deploy_date"] = sanitize_date(ds.deploy_date)
        ds["end_date"] = sanitize_date(ds.end_date)
        ds["drogue_lost_date"] = sanitize_date(ds.drogue_lost_date)
        ds["time"] = sanitize_date(ds.time)
        ds = xr.decode_cf(ds)

        return ds
