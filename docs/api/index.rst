API reference
=============

The public ``ncrf`` API provides a convenience function as well as composable
data, estimator, solver, model, and result objects.

.. currentmodule:: ncrf

High-level interface
--------------------

.. autosummary::
    :toctree: generated/

    fit_ncrf

Fitting pipeline
----------------

.. autosummary::
    :toctree: generated/

    RegressionData
    NCRFEstimator
    CrossValidation

Models and results
------------------

.. autosummary::
    :toctree: generated/

    NCRF
    NCRFResult

Solvers
-------

.. autosummary::
    :toctree: generated/

    Solver
    SolverResult
    ChampLasso

Metrics
-------

.. autosummary::
    :toctree: generated/

    explained_variance
    l2_error
