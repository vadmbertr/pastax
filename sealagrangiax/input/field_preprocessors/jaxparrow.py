from typing import Callable

import jax
import jax.numpy as jnp
import jaxparrow as jpw
import xarray as xr

from .._preprocessor import Preprocessor


class Jaxparrow(Preprocessor):
    def __init__(self, inversion_func: Callable):
        self.id = f"jpw_{inversion_func.__name__}"
        self.inversion_func = inversion_func

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        print(f"Estimating SSC using {self.id}...")

        lat_t = jnp.full((ds.lat_t.size, ds.lon_t.size), ds.lat_t.data.reshape(-1, 1))
        lon_t = jnp.full((ds.lat_t.size, ds.lon_t.size), ds.lon_t.data)
        ssh = jnp.asarray(ds.ssh.data)

        vmap_inversion_func = jax.vmap(
            self.inversion_func,
            in_axes=(0, None, None),
            out_axes=(0, 0, None, None, None, None)
        )
        u_u, v_v, _, lon_u, lat_v, _ = vmap_inversion_func(ssh, lat_t, lon_t)

        ds = xr.Dataset(
            {
                "ssh": (["time", "lat_t", "lon_t"], ssh),
                "u": (["time", "lat_t", "lon_u"], u_u),
                "v": (["time", "lat_v", "lon_t"], v_v)
            },
            coords={
                "time": ds.time.data,
                "lat_t": ds.lat_t.data, "lon_t": ds.lon_t.data,
                "lat_v": lat_v[:, 0], "lon_u": lon_u[0, :]
            }
        )

        return ds


class Geostrophy(Jaxparrow):
    def __init__(self):
        super().__init__(inversion_func=jpw.geostrophy)


class Cyclogeostrophy(Jaxparrow):
    def __init__(self):
        super().__init__(inversion_func=jpw.cyclogeostrophy)
