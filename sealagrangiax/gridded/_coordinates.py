from __future__ import annotations

import equinox as eqx
from jaxtyping import Float, Int, Array

from ._gridded import Coordinate


class Coordinates(eqx.Module):
    time: Coordinate
    latitude: Coordinate
    longitude: Coordinate

    @eqx.filter_jit
    def indices(
        self,
        time: Int[Array, "..."],
        latitude: Float[Array, "..."],
        longitude: Float[Array, "..."]
    ) -> (Int[Array, "..."], Int[Array, "..."], Int[Array, "..."]):
        return self.time.index(time), self.latitude.index(latitude), self.longitude.index(longitude + 180)

    @staticmethod
    @eqx.filter_jit
    def from_arrays(
        time: Int[Array, "time"],
        latitude: Float[Array, "lat"],
        longitude: Float[Array, "lon"]
    ) -> Coordinates:
        t = Coordinate.from_array(time, extrap=True)
        lat = Coordinate.from_array(latitude, extrap=True)
        lon = Coordinate.from_array(longitude + 180, extrap=True, period=360)

        return Coordinates(time=t, latitude=lat, longitude=lon)  # noqa
