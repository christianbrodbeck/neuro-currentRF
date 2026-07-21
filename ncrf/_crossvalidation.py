"""Cross-validation helpers used by the NCRF estimator.

The core model owns fitting and scoring logic, while this module supplies the
execution machinery for sweeping regularization values and splitting time-series
data into train/test windows.
"""

from __future__ import annotations

# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause)

import logging
import os
from dataclasses import dataclass
from math import ceil
from multiprocessing import Pool
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

from eelbrain._config import CONFIG
import numpy as np
from tqdm import tqdm

from ._data import RegressionData
from ._typing import FloatArray

if TYPE_CHECKING:
    from ._model import NCRFEstimator, NCRF
    from ._solvers import Solver

_worker_context: tuple[Callable, tuple] | None = None


@dataclass(frozen=True)
class CrossValidation:
    """Configuration for selecting among solver candidates.

    Parameters
    ----------
    n_splits
        Number of cross-validation folds.
    n_workers
        Number of worker processes, or ``None`` to use the configured default.
    use_es
        Refine ChampLasso selection with the estimation-stability criterion.
    """

    n_splits: int = 3
    n_workers: int | None = None
    use_es: bool = False


def _initialize_worker(score: Callable, *args: object) -> None:
    """Store large shared inputs once per multiprocessing worker."""
    global _worker_context
    if CONFIG['nice']:
        os.nice(CONFIG['nice'])
    _worker_context = score, args


def _score_worker(value: object) -> object:
    if _worker_context is None:
        raise RuntimeError("cross-validation worker was not initialized")
    score, args = _worker_context
    return score(*args, value)


def compute_es_metric(models: Sequence[NCRF], data: RegressionData) -> float:
    """Compute the estimation-stability metric across cross-validation folds.

    Details can be found at:
    Lim, Chinghway, and Bin Yu. "Estimation stability with cross-validation (ESCV)."
    Journal of Computational and Graphical Statistics 25.2 (2016): 464-492.

    Parameters
    ----------
    models
        Fitted fold models from cross-validation.
    data
        Dataset used to compare their predictions.

    Returns
    -------
    float
        Estimation-stability score.
    """
    Y = np.array([
        np.concatenate([
            model._predict_whitened(covariate).ravel()
            for covariate in data.covariates
        ])
        for model in models
    ])
    Y_bar = Y.mean(axis=0)
    VarY = (((Y - Y_bar) ** 2).sum(axis=1)).mean()
    denominator = (Y_bar ** 2).sum()
    return np.inf if denominator <= 0 else VarY / denominator


@dataclass(frozen=True)
class CVResult:
    """Cross-validation scores for one solver candidate.

    Parameters
    ----------
    solver
        Candidate that was evaluated.
    scores
        Mean held-out scores across folds. Always contains the solver-independent
        model metrics (``explained_variance``, ``l2_error``) and
        ``estimation_stability``; solvers add their own through
        :meth:`SolverFit.score`.
    """

    solver: Solver
    scores: dict[str, float]


def _score_candidate(
        estimator: NCRFEstimator,
        data: RegressionData,
        n_splits: int,
        solver: Solver,
) -> CVResult:
    """Fit and score all cross-validation folds for one solver candidate.

    Each fold is fit through the estimator's single-model primitive, then scored
    on its held-out window with the model metrics plus whatever the solver's fit
    contributes.
    """
    d = max(basis.shape[1] for basis in data.basis)
    kf = TimeSeriesSplit(r=0.05, p=n_splits, d=d)
    fold_solver = solver.without_history()
    models = []
    fold_scores = []
    for train, test in kf.split(data.meg[0][0]):
        traindata = data.timeslice(train)
        testdata = data.timeslice(test)
        model, solver_fit = estimator._fit_model(traindata, fold_solver)
        models.append(model)
        fold_scores.append({
            **model.evaluate(testdata, accept_whitening=True),
            **solver_fit.score(estimator.forward, testdata),
        })

    scores = {key: sum(fold[key] for fold in fold_scores) / len(fold_scores) for key in fold_scores[0]}
    estimation_stability = compute_es_metric(models, data)
    scores['estimation_stability'] = 10 if np.isnan(estimation_stability) else estimation_stability
    return CVResult(solver, scores)


def crossvalidate(
        estimator: NCRFEstimator,
        data: RegressionData,
        candidates: Sequence[Solver],
        n_splits: int,
        n_workers: int | None = None,
) -> list[CVResult]:
    """Perform cross-validation over a set of solver candidates.

    Each candidate is fit and scored on the same folds, and the resulting
    :class:`CVResult` objects are returned for the caller to compare.

    Parameters
    ----------
    estimator
        The :class:`NCRF` estimator to validate. It must be picklable so that it
        can be sent to worker processes.
    data
        M/EEG data and the corresponding stimulus variables.
    candidates
        Fixed solver configurations to compare.
    n_splits
        number of folds for cross-validation.
    n_workers
        Number of workers to use for cross-validation.
        ``None`` to use ``cpu_count/2`` (default).
        ``0`` to run without :mod:`multiprocessing`.

    Returns
    -------
    list
        Cross-validation results.
    """
    if n_workers is None:
        n = CONFIG['n_workers'] or 1  # by default this is cpu_count()
        n_workers = ceil(n / 8)

    results = []
    with tqdm(total=len(candidates), desc="Crossvalidation", unit='candidate', unit_scale=True) as prog:
        if n_workers == 0:
            for candidate in candidates:
                results.append(_score_candidate(estimator, data, n_splits, candidate))
                prog.update()
        else:
            with Pool(
                    processes=n_workers,
                    initializer=_initialize_worker,
                    initargs=(_score_candidate, estimator, data, n_splits),
            ) as pool:
                for result in pool.imap_unordered(_score_worker, candidates):
                    results.append(result)
                    prog.update()

    return results


def select_solver(
        estimator: NCRFEstimator,
        data: RegressionData,
        candidates: Sequence[Solver],
        cv: CrossValidation,
) -> tuple[Solver, list[CVResult]]:
    """Cross-validate solver candidates and choose the best one.

    The solver decides how to score, compare and extend its candidates; this
    function only drives the passes.
    """
    logger = logging.getLogger(__name__)
    logger.info('Crossvalidation initiated!')
    cv_results = crossvalidate(estimator, data, candidates, cv.n_splits, cv.n_workers)
    solver = candidates[0].select(cv_results, cv)

    extra = candidates[0].refine(candidates, solver)
    if extra:
        cv_results.extend(crossvalidate(estimator, data, extra, cv.n_splits, cv.n_workers))
        solver = candidates[0].select(cv_results, cv)
    return solver, cv_results


class TimeSeriesSplit:
    """Split contiguous time indices into ordered train/test windows."""

    def __init__(self, r: float = 0.05, p: int = 5, d: int = 100):
        self.ratio = r
        self.p = p
        self.d = d

    def __repr__(self) -> str:
        r, p, d = self.ratio, self.p, self.d
        return f'{type(self).__name__}({r=}, {p=}, {d=})'

    def _iter_part_masks(self, X: Sequence[object] | FloatArray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield boolean masks for each backward-moving validation split."""
        n_v = ceil(self.ratio / (1 + self.ratio) * len(X))
        for i in range(self.p, 0, -1):
            test_mask = np.zeros(len(X), dtype=bool)
            train_mask = np.ones(len(X), dtype=bool)
            train_mask[-(i * n_v + self.d):] = False
            if i == 1:
                test_mask[-i * n_v:] = True
            else:
                test_mask[-i * n_v:-(i - 1) * n_v] = True
            yield train_mask, test_mask

    def split(self, X: Sequence[object] | FloatArray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield integer index arrays for each validation split."""
        indices = np.arange(len(X))
        for (train_mask, test_mask) in self._iter_part_masks(X):
            train_index = indices[train_mask]
            test_index = indices[test_mask]
            yield train_index, test_index
