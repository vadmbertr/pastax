import os

import s3fs

from ._s3filesystem import S3FileSystem


class ZarrStore:
    def __init__(self, data_path: str, zarr_dir: str, filesystem: S3FileSystem = None):
        self.data_path = data_path
        self.zarr_dir = zarr_dir
        self.filesystem = filesystem
        self.zarr_path = os.path.join(self.data_path, self.zarr_dir)
        if self.filesystem is None:
            os.makedirs(self.data_path, exist_ok=True)
            self.store = self.zarr_path
        else:
            self.filesystem.makedirs(self.data_path, exist_ok=True)
            self.store = s3fs.S3Map(root=f"s3://{self.zarr_path}", s3=self.filesystem)

    def exists(self) -> bool:
        if self.filesystem is None:
            return os.path.exists(self.zarr_path)
        else:
            return self.filesystem.exists(self.zarr_path)
