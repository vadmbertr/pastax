from __future__ import annotations
from typing import Dict

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int
import numpy as np
import xarray as xr

from ..utils.unit import degrees_to_radians, EARTH_RADIUS  # noqa
from ._coordinates import Coordinates
from ._gridded import Spatiotemporal


# Defines a Dataset of JAX `Array`s  TODO: special handling for C-grids?
class Dataset(eqx.Module):
    variables: Dict[str, Spatiotemporal]
    is_land: Bool[Array, "lat lon"]
    dx: Float[Array, "lat lon-1"]  # m
    dy: Float[Array, "lat-1 lon"]  # m
    cell_area: Bool[Array, "lat lon"]  # m^2
    coordinates: Coordinates

    # gets nearest indices of the spatio-temporal point `time`, `latitude`, `longitude`
    @eqx.filter_jit
    def indices(
        self,
        time: Int[Array, ""],
        latitude: Float[Array, ""],
        longitude: Float[Array, ""]
    ) -> (Int[Array, "..."], Int[Array, "..."], Int[Array, "..."]):
        return self.coordinates.indices(time, latitude, longitude)

    @eqx.filter_jit
    def interp_temporal(
        self,
        *variables: (str, ...),
        time: Float[Array, "..."]
    ) -> (Float[Array, "... ... ..."], ...):
        return tuple(self.variables[var_name].interp_temporal(time) for var_name in variables)

    @eqx.filter_jit
    def interp_spatial(
        self,
        *variables: (str, ...),
        latitude: Float[Array, "..."],
        longitude: Float[Array, "..."]
    ) -> (Float[Array, "... ... ..."], ...):
        return tuple(
            self.variables[var_name].interp_spatial(latitude, longitude)
            for var_name in variables
        )

    @eqx.filter_jit
    def interp_spatiotemporal(
        self,
        *variables: (str, ...),
        time: Float[Array, "..."],
        latitude: Float[Array, "..."],
        longitude: Float[Array, "..."]
    ) -> (Float[Array, "... ... ..."], ...):
        return tuple(
            self.variables[var_name].interp_spatiotemporal(time, latitude, longitude)
            for var_name in variables
        )

    # gets a neighborhood `t_width`*`x_width`*`x_width` around the spatio-temporal point `time`, `latitude`, `longitude`
    @eqx.filter_jit
    def neighborhood(
        self,
        *variables: (str, ...),
        time: Int[Array, ""],
        latitude: Float[Array, ""],
        longitude: Float[Array, ""],
        t_width: int = 2,
        x_width: int = 7
    ) -> Dataset:
        t_i, lat_i, lon_i = self.indices(time, latitude, longitude)

        from_t_i = t_i - t_width // 2
        from_lat_i = lat_i - x_width // 2
        from_lon_i = lon_i - x_width // 2

        variables = dict(
            (
                var_name,
                jax.lax.dynamic_slice(
                    self.variables[var_name].values,
                    (from_t_i, from_lat_i, from_lon_i),
                    (t_width, x_width, x_width)
                )
            )
            for var_name in variables
        )

        t = jax.lax.dynamic_slice_in_dim(self.coordinates.time.values, from_t_i, t_width)
        lat = jax.lax.dynamic_slice_in_dim(self.coordinates.latitude.values, from_lat_i, x_width)
        lon = jax.lax.dynamic_slice_in_dim(self.coordinates.longitude.values, from_lon_i, x_width) - 180

        return Dataset.from_arrays(variables, t, lat, lon, interpolation_method="linear")

    # construct a `Dataset` from `Array`s of `variables` and coordinates `time`, `latitude`, `longitude`
    @staticmethod
    def from_arrays(
        variables: Dict[str, Float[Array, "time lat lon"]],
        time: Int[Array, "time"],
        latitude: Float[Array, "lat"],
        longitude: Float[Array, "lon"],
        interpolation_method: str
    ) -> Dataset:
        @jax.jit
        def compute_dlatlon(latlon: Float[Array, "latlon"]) -> Float[Array, "latlon-1"]:
            return latlon[1:] - latlon[:-1]

        @jax.jit
        def compute_cell_dlatlon(dright: Float[Array, "latlon-1"]) -> Float[Array, "latlon"]:
            dcentered = (dright[1:] + dright[:-1]) / 2
            dlatlon = jnp.pad(
                dcentered,
                1,
                constant_values=((dright[0] - dcentered[0] / 2) * 2, (dright[-1] - dcentered[-1] / 2) * 2)
            )
            return dlatlon

        @jax.jit
        def dlat_to_meters(_dlat: Float[Array, "lat-1"]) -> Float[Array, "lat-1 lon"]:
            return jnp.einsum("i,j->ij", degrees_to_radians(_dlat), jnp.full_like(longitude, EARTH_RADIUS))

        @jax.jit
        def dlon_to_meters(_dlon: Float[Array, "lon-1"]) -> Float[Array, "lat lon-1"]:
            return jnp.einsum(
                "i,j->ij",
                jnp.cos(degrees_to_radians(latitude)) * EARTH_RADIUS, degrees_to_radians(_dlon)
            )

        is_land = jnp.zeros((latitude.size, longitude.size), dtype=bool)

        for variable in variables.values():
            _is_land = jnp.isnan(variable).sum(axis=0, dtype=bool)
            is_land = jnp.logical_or(is_land, _is_land)

        for var_name in variables:
            variable = variables[var_name]
            variable = jnp.where(is_land, 0, variable)
            variables[var_name] = variable

        coordinates = Coordinates.from_arrays(time, latitude, longitude)

        variables = dict(
            (
                variable_name,
                Spatiotemporal.from_array(
                    values,
                    coordinates.time.values, coordinates.latitude.values, coordinates.longitude.values,
                    interpolation_method=interpolation_method
                )
            ) for variable_name, values in variables.items()
        )

        dlat = compute_dlatlon(latitude)  # degrees
        dlon = compute_dlatlon(longitude)  # degrees

        cell_dlat = compute_cell_dlatlon(dlat)  # degrees
        cell_dlon = compute_cell_dlatlon(dlon)  # degrees

        dlat = dlat_to_meters(dlat)  # meters
        dlon = dlon_to_meters(dlon)  # meters

        cell_dlat = dlat_to_meters(cell_dlat)  # meters
        cell_dlon = dlon_to_meters(cell_dlon)  # meters
        cell_area = cell_dlat * cell_dlon  # m^2

        return Dataset(variables=variables, is_land=is_land, dx=dlon, dy=dlat, cell_area=cell_area, coordinates=coordinates)  # noqa

    # construct a `Dataset` of `variables` and `coordinates` from a `xarray.Dataset`
    @staticmethod
    def from_xarray(
            dataset: xr.Dataset,
            variables: Dict[str, str],  # to -> from
            coordinates: Dict[str, str],  # to -> from
            interpolation_method: str
    ) -> Dataset:
        variables = dict(
            (to_name, jnp.asarray(dataset[from_name].values)) for to_name, from_name in variables.items()
        )

        t = jnp.asarray(dataset[coordinates["time"]].values.astype("datetime64[s]"), dtype=int)
        lat = jnp.asarray(dataset[coordinates["latitude"]].values)
        lon = jnp.asarray(dataset[coordinates["longitude"]].values)

        return Dataset.from_arrays(variables, t, lat, lon, interpolation_method)

    # returns a `Dataset` as a `xarray.Dataset`
    def to_xarray(self) -> xr.Dataset:
        dataset = xr.Dataset(
            data_vars=dict(
                (var_name, (["time", "latitude", "longitude"], var.values)) for var_name, var in self.variables.items()
            ),
            coords=dict(
                time=np.asarray(self.coordinates.time.values, dtype="datetime64[s]"),
                latitude=self.coordinates.latitude.values,
                longitude=self.coordinates.longitude.values
            )
        )

        return dataset
