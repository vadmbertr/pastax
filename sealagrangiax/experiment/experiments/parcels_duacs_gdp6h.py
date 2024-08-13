import jax

from ...input import Duacs, GDP6hLocal, JpwCycloPreproc, JpwGeosPreproc, Preprocessing
from ...simulator import DeterministicParcels
from .._experiment import Experiment


class ParcelsDuacsGDP6h(Experiment):
    def __init__(
            self,
            duacs_preproc: Preprocessing = None,
            gdp6h_preproc: Preprocessing = None,
            experiment_id: str = "parcels_duacs_gdp6h",
            data_path: str = "data"
    ):
        ssc_field = Duacs(preprocessing=duacs_preproc)
        simulator = DeterministicParcels({"ssc": ssc_field})
        drifters = GDP6hLocal(preprocessing=gdp6h_preproc, data_path=data_path)
        super().__init__(simulator, drifters, experiment_id=experiment_id, data_path=data_path)

    def _simulate_and_evaluate_all(self, n_samples: int = None, key: jax.random.PRNGKey = None, plot: bool = False):
        for trajectory_idx in range(len(self.drifters)):
            self._simulate_and_evaluate_one(trajectory_idx, plot=plot)


class ParcelsCycloGDP6h(ParcelsDuacsGDP6h):
    def __init__(self, gdp6h_preproc: Preprocessing = None, data_path: str = "data"):
        super().__init__(
            duacs_preproc=Preprocessing([JpwCycloPreproc()]),
            gdp6h_preproc=gdp6h_preproc,
            experiment_id="parcels_cyclo_gdp6h",
            data_path=data_path
        )


class ParcelsGeosGDP6h(ParcelsDuacsGDP6h):
    def __init__(self, gdp6h_preproc: Preprocessing = None, data_path: str = "data"):
        super().__init__(
            duacs_preproc=Preprocessing([JpwGeosPreproc()]),
            gdp6h_preproc=gdp6h_preproc,
            experiment_id="parcels_geos_gdp6h",
            data_path=data_path
        )
