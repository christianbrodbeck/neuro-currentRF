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
from dataclasses import dataclass, replace
from math import ceil
from multiprocessing import Pool
from operator import attrgetter
from typing import TYPE_CHECKING, Callable, Iterator, List, Sequence

from eelbrain._config import CONFIG
import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks
from tqdm import tqdm

from ._data import RegressionData
from ._metrics import l2_error
from ._solvers import ChampLasso

if TYPE_CHECKING:
    from ._model import NCRF, NCRFModel

FloatArray = npt.NDArray[np.float64]
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


def compute_es_metric(models: Sequence[NCRFModel], data: RegressionData) -> float:
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
    weighted_l2_error
        Mean held-out weighted L2 error.
    estimation_stability
        Prediction stability across folds.
    cross_fit
        Mean held-out ChampLasso objective.
    l2_error
        Mean held-out unweighted L2 error.
    """

    solver: ChampLasso
    weighted_l2_error: float
    estimation_stability: float
    cross_fit: float
    l2_error: float


def _score_candidate(
        estimator: NCRF,
        data: RegressionData,
        n_splits: int,
        solver: ChampLasso,
) -> CVResult:
    """Fit and score all cross-validation folds for one solver candidate.

    Each fold is fit through the estimator's single-model primitive, then
    scored on its held-out window.
    """
    d = max(basis.shape[1] for basis in data.basis)
    kf = TimeSeriesSplit(r=0.05, p=n_splits, d=d)
    models = []
    weighted_l2 = []
    cross_fit = []
    l2 = []
    for train, test in kf.split(data.meg[0][0]):
        traindata = data.timeslice(train)
        testdata = data.timeslice(test)
        fold_solver = replace(
            solver,
            store_objective=False,
            store_residual=False,
            store_theta=False,
            store_gamma=False,
            store_sigma_b=False,
        )
        model, solver_fit = estimator._fit_model(traindata, fold_solver)
        models.append(model)
        obj, wl2 = solver_fit.evaluate_objective(
            estimator.forward, testdata, True,
        )
        weighted_l2.append(wl2)
        cross_fit.append(obj)
        l2.append(l2_error(model, testdata, accept_whitening=True))

    estimation_stability = compute_es_metric(models, data)
    return CVResult(
        solver,
        sum(weighted_l2) / len(weighted_l2),
        10 if np.isnan(estimation_stability) else estimation_stability,
        sum(cross_fit) / len(cross_fit),
        sum(l2) / len(l2),
    )


def crossvalidate(
        estimator: NCRF,
        data: RegressionData,
        candidates: Sequence[ChampLasso],
        n_splits: int,
        n_workers: int | None = None,
) -> List[CVResult]:
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
        Fixed ChampLasso configurations to compare.
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


def select_best_solver(cv_results: Sequence[CVResult], criterion: str = 'cross-fit') -> ChampLasso:
    """Pick the best solver from cross-validation results by the given criterion.

    Parameters
    ----------
    cv_results
        Results to choose from.
    criterion
        Criterion for best fit. Possible values:

        - ``'cross-fit'``: The smallest cross-fit value (default)
        - ``'l2'``: The smallest l2 error
        - ``'l2/mu'``: The local minimum in the l2 error with smallest trf (largest mu)
    """
    if criterion == 'cross-fit':
        return min(cv_results, key=attrgetter('cross_fit')).solver
    elif criterion == 'l2':
        return min(cv_results, key=attrgetter('l2_error')).solver
    elif criterion == 'l2/mu':
        results = sorted(cv_results, key=attrgetter('solver.mu'))
        peaks, _ = find_peaks([-result.l2_error for result in results])  # find local minima
        if len(peaks) > 0:
            # higher mu -> smaller trf
            return max((results[peak] for peak in peaks), key=attrgetter('solver.mu')).solver
        return min(results, key=attrgetter('l2_error')).solver
    else:
        raise ValueError(f'criterion={criterion}')


def _extend_mu_grid(candidates: Sequence[ChampLasso], best_solver: ChampLasso) -> tuple[ChampLasso, ...]:
    """Return one additional decade when the best candidate is on a grid boundary."""
    mus = [candidate.mu for candidate in candidates]
    best_mu = best_solver.mu
    if best_mu == min(mus):
        new_mus = np.logspace(np.log10(best_mu) - 1, np.log10(best_mu), 4)[:-1]
    elif best_mu == max(mus):
        new_mus = np.logspace(np.log10(best_mu), np.log10(best_mu) + 1, 4)[1:]
    else:
        return ()
    return tuple(replace(best_solver, mu=float(mu)) for mu in new_mus)


def _select_es_solver(cv_results: Sequence[CVResult], minimum_mu: float) -> ChampLasso | None:
    """Return the solver at the first ES local minimum above ``minimum_mu``."""
    results = sorted(cv_results, key=attrgetter('solver.mu'))
    for i, result in enumerate(results[:-1]):
        if result.solver.mu < minimum_mu:
            continue
        if result.estimation_stability < results[i + 1].estimation_stability:
            return result.solver
    return None


def search_param(
        estimator: NCRF,
        data: RegressionData,
        candidates: Sequence[ChampLasso],
        cv: CrossValidation,
) -> tuple[ChampLasso, List[CVResult]]:
    """Cross-validate ChampLasso candidates and choose the regularization parameter.

    Extends the candidate grid by a decade if the best value lands on a boundary,
    and optionally refines the choice with the estimation-stability criterion.
    """
    logger = logging.getLogger(__name__)
    logger.info('Crossvalidation initiated!')
    cv_results = crossvalidate(
        estimator, data, candidates, cv.n_splits, cv.n_workers,
    )
    solver = select_best_solver(cv_results, 'cross-fit')
    new_candidates = _extend_mu_grid(candidates, solver)

    if new_candidates:
        direction = 'left' if new_candidates[-1].mu < solver.mu else 'right'
        logger.info(f'CVmu is {solver.mu}: extending range of mu towards {direction}')
        cv_results.extend(crossvalidate(
            estimator, data, new_candidates, cv.n_splits, cv.n_workers,
        ))
        solver = select_best_solver(cv_results, 'cross-fit')

    if cv.use_es:
        if solver.mu == max(result.solver.mu for result in cv_results):
            logger.info(f'\nCVmu is {solver.mu}: could not find mu based on estimation stability criterion\nContinuing with cross-validation only.')
        else:
            es_solver = _select_es_solver(cv_results, solver.mu)
            if es_solver is None:
                logger.warning('\nNo ES minima found: could not find mu based on estimation stability criterion.\nContinuing with cross-validation only.')
            else:
                solver = es_solver
    return solver, cv_results


class TimeSeriesSplit:
    """Split contiguous time indices into ordered train/test windows."""

    def __init__(self, r: float = 0.05, p: int = 5, d: int = 100):
        self.ratio = r
        self.p = p
        self.d = d

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
