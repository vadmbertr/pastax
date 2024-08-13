import jax.random as jrd

from ...input import Duacs, GDP6hLocal, Preprocessing
from ...simulator import SmagorinskyDiffrax
from .._experiment import Experiment


class DiffraxSmagDuacsGDP6h(Experiment):
    def __init__(
            self,
            gdp6h_preproc: Preprocessing = None,
            experiment_id: str = "diffrax_smag_duacs_gdp6h",
            data_path: str = "data"
    ):
        ssc_field = Duacs()
        simulator = SmagorinskyDiffrax({"ssc": ssc_field})
        drifters = GDP6hLocal(preprocessing=gdp6h_preproc, data_path=data_path)
        super().__init__(simulator, drifters, experiment_id=experiment_id, data_path=data_path)

    def _simulate_and_evaluate_all(self, n_samples: int = 1, key: jrd.PRNGKey = None, plot: bool = False):
        keys = jrd.split(key, n_samples)
        for trajectory_idx in range(len(self.drifters)):
            self._simulate_and_evaluate_one(trajectory_idx, n_samples=n_samples, key=keys[trajectory_idx], plot=plot)
