"""Public package surface for the NCRF fitting pipeline.

The package is organized around a small top-level API: :func:`fit_ncrf`
coordinates input normalization and model fitting, while :class:`NCRF`,
:class:`Solver`, and :class:`RegressionData` expose the lower-level workflow.
"""

from ._data import RegressionData
from ._crossvalidation import CrossValidation
from ._model import NCRFEstimator, NCRF, NCRFResult
from ._metrics import explained_variance, l2_error
from ._solvers import ChampLasso, Solver, SolverResult
from ._ncrf import fit_ncrf
