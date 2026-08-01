"""Public API for fitting and applying neuro-current response functions.

:func:`fit_ncrf` provides the complete convenience workflow. The component API
separates prepared :class:`RegressionData`, the :class:`NCRFEstimator` fitting
orchestrator, pluggable :class:`Solver` configurations, the reusable fitted
:class:`NCRF`, and the accompanying :class:`NCRFFit` report.
"""

from ._data import RegressionData
from ._crossvalidation import CrossValidation, CVResult
from ._model import NCRFEstimator, NCRF, NCRFFit
from ._metrics import explained_variance, l2_error
from ._solvers import ChampLasso, ChampLassoFit, Solver, SolverFit
from ._ncrf import fit_ncrf
