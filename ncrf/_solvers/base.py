"""The solver contract.

A solver is immutable configuration that estimates NCRF weights (:meth:`Solver.solve`).
Solvers that expose more than one candidate configuration (:meth:`Solver.candidates`)
are selected by cross-validation, which is driven entirely through the hooks below:

- :meth:`SolverResult.score` contributes solver-specific scores.
- :attr:`Solver.criterion` names the score to minimize.
- :meth:`Solver.select` picks the winner, :meth:`Solver.refine` may extend the search.

Every hook has a working default, so a new solver only needs :meth:`solve`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from eelbrain import fmtxt

from .._repr import _count_repr
from .._typing import FloatArray

if TYPE_CHECKING:
    from .._crossvalidation import CrossValidation, CVResult
    from .._data import RegressionData
    from .._forward import ForwardModel


@dataclass(frozen=True, repr=False)
class SolverResult:
    """Fitted state from one solver execution.

    The generic result contains the coefficient matrix consumed by
    :class:`~ncrf.NCRF`. Concrete solvers can add optimizer-specific state,
    diagnostics, and scores without coupling those details to the predictive
    model.

    Parameters
    ----------
    theta
        Fitted source-space coefficients over the regression design basis.
    """

    theta: FloatArray

    def __repr__(self) -> str:
        n_components, n_basis = self.theta.shape
        return f"<{type(self).__name__}: {_count_repr(n_components, 'source component')}, {_count_repr(n_basis, 'basis coefficient')}>"

    def score(
            self,
            forward: ForwardModel,
            data: RegressionData,
    ) -> dict[str, float]:
        """Solver-specific scores for ``data``, e.g. the training objective.

        Merged with the solver-independent model metrics wherever a fit is
        scored: on the training data in :attr:`NCRFResult.scores`, and per fold
        in ``CVResult.scores``. Keys must not collide with the metric names.
        """
        return {}


class Solver(ABC):
    """Configuration contract for an algorithm that estimates NCRF weights.

    :class:`~ncrf.NCRFEstimator` asks a solver for fixed candidate
    configurations, cross-validates them when necessary, and calls :meth:`solve`
    for the final fit. Implementations can override the selection and refinement
    hooks while returning a common :class:`SolverResult` interface.
    """

    #: Key in :attr:`CVResult.scores` minimized when selecting among candidates.
    criterion: str = 'l2_error'

    def candidates(
            self,
            forward: ForwardModel,
            data: RegressionData,
    ) -> tuple[Solver, ...]:
        """Return fixed configurations to compare before fitting."""
        return (self,)

    @abstractmethod
    def solve(
            self,
            forward: ForwardModel,
            data: RegressionData,
            *,
            verbose: bool = False,
    ) -> SolverResult:
        """Estimate source-space NCRF weights for prepared, whitened data."""

    def without_history(self) -> Solver:
        """Return this configuration with per-iteration storage disabled.

        Used for cross-validation folds, whose history is discarded.
        """
        return self

    def select(
            self,
            cv_results: Sequence[CVResult],
            cv: CrossValidation,
    ) -> Solver:
        """Choose the best candidate from cross-validation results."""
        return min(cv_results, key=lambda result: result.scores[self.criterion]).solver

    def refine(
            self,
            candidates: Sequence[Solver],
            best: Solver,
    ) -> tuple[Solver, ...]:
        """Additional candidates to score after a first selection pass.

        Returning ``()`` (the default) ends the search after one pass.
        """
        return ()

    def cv_table(
            self,
            cv_results: Sequence[CVResult],
            selected: Solver,
    ) -> fmtxt.Table:
        """Summarize cross-validation scores in a table."""
        keys = sorted(cv_results[0].scores)
        table = fmtxt.Table('l' * (len(keys) + 1))
        table.cells('solver', *keys)
        table.midrule()
        for result in cv_results:
            marker = '*' if result.solver == selected else ''
            table.cell(f'{result.solver!r}{marker}')
            for key in keys:
                table.cell(fmtxt.stat(result.scores[key], fmt='%.5f'))
        return table
