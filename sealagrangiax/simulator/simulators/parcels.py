from datetime import datetime, timedelta
import shutil
from typing import Dict

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int
import numpy as np
from parcels import FieldSet, JITParticle, ParticleSet, AdvectionRK4
import xarray as xr

from ...input import Field
from ...trajectory import Location, Trajectory
from ...utils.unit import time_in_seconds  # noqa
from .._simulator import Simulator


class DeterministicParcels(Simulator):
    def __init__(self, fields: Dict[str, Field], simulator_id="deterministic_parcels", data_path: str = "data"):
        super().__init__(fields, simulator_id=simulator_id)
        self.data_path = data_path

    def __get_fieldset(self, x0: Location, t0: Int[Array, ""], ts: Int[Array, "time-1"]) -> FieldSet:
        min_time, max_time, min_point, max_point = self._get_minmax(x0, t0, ts)

        self.fields["ssc"].load_data(
            min_time.item(), max_time.item(),
            min_point.latitude.item(), max_point.latitude.item(),
            min_point.longitude.item(), max_point.longitude.item()
        )

        ds = self.fields["ssc"].dataset

        dimensions = {"lon": "lon_t", "lat": "lat_t", "time": "time"}
        interp_method = "linear"
        if self.fields["ssc"].is_cgrid:
            dimensions = {"lon": "lon_u", "lat": "lat_v", "time": "time"}
            interp_method = "cgrid_velocity"
            # we need to "trick" Parcels...
            ds = ds.copy()
            ds["u"] = (("time", "lat_v", "lon_u"), ds["u"].data)
            ds["v"] = (("time", "lat_v", "lon_u"), ds["v"].data)

        # Parcels replaces nan with 0. ; mem-copy of underlying arrays
        ds["u"] = ((dimensions["time"], dimensions["lat"], dimensions["lon"]), ds["u"].data.copy())
        ds["v"] = ((dimensions["time"], dimensions["lat"], dimensions["lon"]), ds["v"].data.copy())

        fieldset = FieldSet.from_xarray_dataset(
            ds,
            variables={"U": "u", "V": "v"},
            dimensions=dimensions,
            interp_method=interp_method
        )

        return fieldset

    @staticmethod
    def __get_drifter(ssc: FieldSet, x0: Location, t0: Int[Array, ""]) -> ParticleSet:
        return ParticleSet.from_list(
            fieldset=ssc,
            pclass=JITParticle,  # noqa
            lon=[x0.longitude],
            lat=[x0.latitude],
            time=[datetime.fromtimestamp(t0.item())]
        )

    def __call__(
            self,
            x0: Location,
            t0: Int[Array, ""],
            ts: Int[Array, "time-1"],
            n_samples: int = None,
            key: jax.random.PRNGKey = None
    ) -> Trajectory:
        print("Simulating drift using OceanParcels...")

        # Parcels stuff...
        ssc = self.__get_fieldset(x0, t0, ts)
        drifter = self.__get_drifter(ssc, x0, t0)

        # Parcels outputs to a zarr store
        output_file = f"{self.data_path}/tmp/parcels_{int(np.datetime64('today', 'ns'))}.zarr"
        output_dt = timedelta(seconds=(ts[1] - ts[0]).item())
        particle_file = drifter.ParticleFile(name=output_file, outputdt=output_dt)
        t1 = datetime.fromtimestamp(ts[-1].item()) + output_dt

        drifter.execute(
            AdvectionRK4,
            endtime=t1,
            dt=timedelta(minutes=5),
            output_file=particle_file
        )

        dataset = xr.open_zarr(output_file)
        simulated_trajectory = Trajectory(
            jnp.stack((dataset.lat.values.flatten(), dataset.lon.values.flatten())).T,
            time_in_seconds(dataset.time.data.flatten())
        )

        shutil.rmtree(output_file)

        return simulated_trajectory
