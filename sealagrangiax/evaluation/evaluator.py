from ..trajectory import Trajectory, TrajectoryEnsemble
from ._evaluation import Evaluation
from ._evaluator import Evaluator


class PairEvaluator(Evaluator):
    """
    Class for evaluating a simulated trajectory using a set of predefined metrics.

    Methods
    -------
    __call__(self, reference_trajectory: Trajectory, simulated_trajectory: Trajectory) -> Evaluation
        Evaluates the `simulated_trajectory` against the `reference_trajectory` using the defined metrics.
    """
    
    def __call__(self, reference_trajectory: Trajectory, simulated_trajectory: Trajectory) -> Evaluation:
        """
        Evaluates the `simulated_trajectory` against the `reference_trajectory` using the defined metrics.
        
        Parameters
        ----------
            reference_trajectory (Trajectory): The reference trajectory to compare against.
            simulated_trajectory (Trajectory): The simulated trajectory to be evaluated.
        
        Returns
        -------
            Evaluation: The result of the evaluation.
        """
        metrics = {}
        for metric in self.metrics:
            metric_fun = metric.metric_fun
            metrics[metric_fun] = getattr(reference_trajectory, metric_fun)(simulated_trajectory)

        return Evaluation(metrics)


class EnsembleEvaluator(Evaluator):
    """
    Class for evaluating an ensemble of simulated trajectories using a set of predefined metrics.

    Methods
    -------
    __call__(self, reference_trajectory: Trajectory, simulated_trajectory: TrajectoryEnsemble) -> Evaluation
        Evaluates the `simulated_trajectories` ensemble against the `reference_trajectory` using the defined metrics.
    """

    def __call__(self, reference_trajectory: Trajectory, simulated_trajectories: TrajectoryEnsemble) -> Evaluation:
        """
        Evaluates the `simulated_trajectories` ensemble against the `reference_trajectory` using the defined metrics.
        
        Parameters
        ----------
        reference_trajectory (Trajectory)
            The reference trajectory to compare against.
        simulated_trajectory (Trajectory)
            The simulated ensemble of trajectories to be evaluated.
        
        Returns
        -------
        Evaluation
            The result of the evaluation.
        """
        metrics = {}
        for metric in self.metrics:
            metric_fun = metric.metric_fun

            ensemble = getattr(simulated_trajectories, metric_fun)(reference_trajectory)
            ensemble = getattr(simulated_trajectories, metric_fun)(reference_trajectory)

            crps = simulated_trajectories.crps(
                reference_trajectory, distance_func=getattr(Trajectory, metric_fun)
            ).attach_name("CRPS")

            mean = ensemble.mean(axis=0).attach_name("Mean")

            metrics[metric_fun] = (ensemble, crps, mean)

        return Evaluation(metrics)
