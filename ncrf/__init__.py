"""Public API for fitting and applying neuro-current response functions.

:func:`fit_ncrf` provides the complete convenience workflow. The component API
separates prepared :class:`RegressionData`, the :class:`NCRFEstimator` fitting
orchestrator, pluggable :class:`Solver` configurations, the reusable fitted
:class:`NCRF`, and the accompanying :class:`NCRFResult` report.
"""

from ._data import RegressionData
from ._crossvalidation import CrossValidation
from ._model import NCRFEstimator, NCRF, NCRFResult
from ._metrics import explained_variance, l2_error
from ._solvers import ChampLasso, ChampLassoResult, Solver, SolverResult
from ._ncrf import fit_ncrf
