import xarray as xr

from ._preprocessor import Preprocessor


class Preprocessing:
    def __init__(self, preprocessors: [Preprocessor, ...]):
        self.id = self.__create_id(preprocessors)
        self.preprocessors = preprocessors

    @staticmethod
    def __create_id(preprocessors: [Preprocessor, ...]) -> str:
        preprocessing_id = ""

        for preprocessor in preprocessors:
            preprocessing_id += type(preprocessor).__name__

        return preprocessing_id

    def __call__(self, ds: xr.Dataset,  *args, **kwargs):
        for preprocessor in self.preprocessors:
            ds = preprocessor(ds, *args, **kwargs)

        return ds
