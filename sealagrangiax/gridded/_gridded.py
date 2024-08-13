from __future__ import annotations
from typing import Any

import equinox as eqx
import interpax as inx
import jax.numpy as npx
from jaxtyping import Float, Int, Array


class Gridded(eqx.Module):
    values: Float[Array, "..."] | Int[Array, "..."]

    def __getitem__(self, item: Any) -> Float[Array, "..."] | Int[Array, "..."]:
        return self.values.__getitem__(item)


class Coordinate(Gridded):
    values: Float[Array, "dim"]  # only handles 1D coordinates, i.e. rectilinear grids
    indices: inx.Interpolator1D

    @eqx.filter_jit
    def index(self, query: Float[Array, "..."]) -> Int[Array, "..."]:  # nearest index interpolation
        return self.indices(query).astype(int)

    @staticmethod
    @eqx.filter_jit
    def from_array(
        values: Float[Array, "coord"],
        **interpolator_kwargs: Any
    ) -> Coordinate:
        interpolator_kwargs["method"] = "nearest"
        indices = inx.Interpolator1D(values, npx.arange(values.size), **interpolator_kwargs)

        return Coordinate(values=values, indices=indices)  # noqa


class Spatial(Gridded):
    values: Float[Array, "dim1 dim2"]
    spatial_field: inx.Interpolator2D

    @eqx.filter_jit
    def interp_spatial(
        self,
        latitude: Float[Array, "..."],
        longitude: Float[Array, "..."]
    ) -> Float[Array, "... ..."]:
        longitude += 180  # circular domain

        return self.spatial_field(latitude, longitude)

    @staticmethod
    @eqx.filter_jit
    def from_array(
        values: Float[Array, "dim1 dim2"],
        latitude: Float[Array, "dim1"],
        longitude: Float[Array, "dim2"],
        interpolation_method: str
    ) -> Spatial:
        spatial_field = inx.Interpolator2D(
            latitude, longitude,
            values,
            method=interpolation_method, extrap=True, period=(None, 360)
        )

        return Spatial(values=values, spatial_field=spatial_field)  # noqa


class Spatiotemporal(Gridded):
    values: Float[Array, "dim1 dim2 dim3"]
    temporal_field: inx.Interpolator1D
    spatial_field: inx.Interpolator2D
    spatiotemporal_field: inx.Interpolator3D

    @eqx.filter_jit
    def interp_temporal(self, tq: Float[Array, "..."]) -> Float[Array, "... ... ..."]:
        return self.temporal_field(tq)

    @eqx.filter_jit
    def interp_spatial(
        self,
        latitude: Float[Array, "..."],
        longitude: Float[Array, "..."]
    ) -> Float[Array, "... ... ..."]:
        longitude += 180  # circular domain

        return npx.moveaxis(self.spatial_field(latitude, longitude), -1, 0)

    @eqx.filter_jit
    def interp_spatiotemporal(
        self,
        time: Float[Array, "..."],
        latitude: Float[Array, "..."],
        longitude: Float[Array, "..."]
    ) -> Float[Array, "... ... ..."]:
        longitude += 180  # circular domain

        return self.spatiotemporal_field(time, latitude, longitude)

    @staticmethod
    @eqx.filter_jit
    def from_array(
        values: Float[Array, "dim1 dim2 dim3"],
        time: Float[Array, "dim1"],
        latitude: Float[Array, "dim2"],
        longitude: Float[Array, "dim3"],
        interpolation_method: str
    ) -> Spatiotemporal:
        temporal_field = inx.Interpolator1D(
            time,
            values,
            method=interpolation_method, extrap=True
        )
        spatial_field = inx.Interpolator2D(
            latitude, longitude,
            npx.moveaxis(values, 0, -1),
            method=interpolation_method, extrap=True, period=(None, 360)
        )
        spatiotemporal_field = inx.Interpolator3D(
            time, latitude, longitude,
            values,
            method=interpolation_method, extrap=True, period=(None, None, 360)
        )

        return Spatiotemporal(
            values=values,                              # noqa
            temporal_field=temporal_field,              # noqa
            spatial_field=spatial_field,                # noqa
            spatiotemporal_field=spatiotemporal_field,  # noqa
        )

