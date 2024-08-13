import xarray as xr

from ..trajectory import Trajectory
from ._preprocessing import Preprocessing


class Drifter:
    id: str = ""

    def __init__(self, preprocessing: Preprocessing = None):
        self.dataset = self._load_data(preprocessing)

    def _do_load_data(self) -> xr.Dataset:
        raise NotImplementedError

    @staticmethod
    def _apply_preprocessing(ds: xr.Dataset, preprocessing: Preprocessing) -> xr.Dataset:
        if preprocessing is not None:
            ds = preprocessing(ds)

        return ds

    def _load_data(self, preprocessing: Preprocessing) -> xr.Dataset:
        ds = self._do_load_data()
        ds = self._apply_preprocessing(ds, preprocessing)

        return ds

    def __len__(self) -> int:
        if self.dataset is None:
            return 0
        else:
            return len(self.dataset.traj)

    def __getitem__(self, item) -> Trajectory:
        if item >= len(self):
            raise IndexError()

        return self.get_trajectory(item)

    def get_trajectory(self, drifter_idx: int = None) -> Trajectory:
        raise NotImplementedError()
