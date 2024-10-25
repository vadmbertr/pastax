"""
This package provides a set of tools for building and evaluating 
**P**arameterizable **A**uto-differentiable **S**imulators of ocean **T**rajectories in j**AX**.
"""


from ._version import version as __version__
from .evaluation import __all__ as evaluation_all
from .grid import __all__ as grid_all
from .simulator import __all__ as simulator_all
from .trajectory import __all__ as trajectory_all
from .utils import __all__ as utils_all


__all__ = ["__version__", ], evaluation_all + grid_all + simulator_all + trajectory_all + utils_all
