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
from math import ceil
from multiprocessing import Pool
from operator import attrgetter
from typing import TYPE_CHECKING, Iterator, List, Sequence

from eelbrain._config import CONFIG
import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks
from tqdm import tqdm

from ._solver import find_mu_range

if TYPE_CHECKING:
    from ._model import NCRF, NCRFModel, RegressionData

FloatArray = npt.NDArray[np.float64]
_worker_context: tuple[NCRF, RegressionData, int, float] | None = None


def eval_l2(model: NCRFModel, data: RegressionData) -> float:
    """Unweighted L2 prediction error of a fitted model, used to score CV folds."""
    data = model._whiten(data, accept_whitening=True)
    l2 = 0
    for meg, covariate in data:
        y = meg - model._predict_whitened(covariate)
        l2 += 0.5 * (y ** 2).sum()
    return l2 / len(data)


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
    Y = []
    for model in models:
        y = np.empty(0)
        for trial in range(len(data)):
            y = np.append(y, model._predict_whitened(data.covariates[trial]))
        Y.append(y)
    Y = np.array(Y)
    Y_bar = Y.mean(axis=0)
    VarY = (((Y - Y_bar) ** 2).sum(axis=1)).mean()
    if (Y_bar ** 2).sum() <= 0:
        return np.inf
    else:
        return VarY / (Y_bar ** 2).sum()


class CVResult:
    """Cross-validation results

    Parameters
    ----------
    mu
        Optimal ``mu`` parameter.
    weighted_l2_error
        self explanatory
    estimation_stability
        self explanatory
    cross_fit
        self explanatory
    l2_error
        L2 error from the optimal ``mu``.
    """

    def __init__(
            self, mu: float,
            weighted_l2_error: float,
            estimation_stability: float,
            cross_fit: float,
            l2_error: float,
    ):
        self.mu = mu
        self.weighted_l2_error = weighted_l2_error
        self.estimation_stability = 10 if np.isnan(estimation_stability) else estimation_stability  # replace Nan values with a big number
        self.cross_fit = cross_fit
        self.l2_error = l2_error


def _score_mu(
        estimator: NCRF,
        data: RegressionData,
        n_splits: int,
        tol: float,
        mu: float,
) -> CVResult:
    """Fit and score all cross-validation folds for one regularization value.

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
        model = estimator._fit_model(traindata, mu, tol)
        models.append(model)
        obj, wl2 = model.eval_obj(testdata, True, accept_whitening=True)
        weighted_l2.append(wl2)
        cross_fit.append(obj)
        l2.append(eval_l2(model, testdata))

    return CVResult(
        mu,
        sum(weighted_l2) / len(weighted_l2),
        compute_es_metric(models, data),
        sum(cross_fit) / len(cross_fit),
        sum(l2) / len(l2),
    )


def _initialize_worker(
        estimator: NCRF,
        data: RegressionData,
        n_splits: int,
        tol: float,
) -> None:
    """Initialize one worker with the shared CV inputs."""
    global _worker_context
    if CONFIG['nice']:
        os.nice(CONFIG['nice'])
    _worker_context = estimator, data, n_splits, tol


def _score_worker(mu: float) -> CVResult:
    """Score one regularization value using the current worker's CV inputs."""
    if _worker_context is None:
        raise RuntimeError("cross-validation worker was not initialized")
    estimator, data, n_splits, tol = _worker_context
    return _score_mu(estimator, data, n_splits, tol, mu)


def crossvalidate(
        estimator: NCRF,
        data: RegressionData,
        mus: Sequence[float],
        tol: float,
        n_splits: int,
        n_workers: int = None,
) -> List[CVResult]:
    """Perform cross-validation over a set of regularization values.

    For each regularizing weight in ``mus`` the folds are fit and scored by
    :func:`_score_mu`, and the resulting :class:`CVResult` objects are returned
    for the caller to compare.

    Parameters
    ----------
    estimator
        The :class:`NCRF` estimator to validate. It must be picklable so that it
        can be sent to worker processes.
    data
        M/EEG data and the corresponding stimulus variables.
    mus
        The range of the regularizing weights to test.
    tol
        Tolerance parameter. Decides when to stop outer iterations.
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
    with tqdm(total=len(mus), desc="Crossvalidation", unit='mu', unit_scale=True) as prog:
        if n_workers == 0:
            for mu in mus:
                results.append(_score_mu(estimator, data, n_splits, tol, mu))
                prog.update()
        else:
            with Pool(
                    processes=n_workers,
                    initializer=_initialize_worker,
                    initargs=(estimator, data, n_splits, tol),
            ) as pool:
                for result in pool.imap_unordered(_score_worker, mus):
                    results.append(result)
                    prog.update()

    return results


def select_best_mu(cv_results: List[CVResult], criterion: str = 'cross-fit') -> float:
    """Pick the best ``mu`` from cross-validation results by the given criterion.

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
        return min(cv_results, key=attrgetter('cross_fit')).mu
    elif criterion == 'l2':
        return min(cv_results, key=attrgetter('l2_error')).mu
    elif criterion == 'l2/mu':
        results = sorted(cv_results, key=attrgetter('mu'))
        peaks, _ = find_peaks([-result.l2_error for result in results])  # find local minima
        if len(peaks) > 0:
            # higher mu -> smaller trf
            return max((results[peak] for peak in peaks), key=attrgetter('mu')).mu
        return min(results, key=attrgetter('l2_error')).mu
    else:
        raise ValueError(f'criterion={criterion}')


def _extend_mu_grid(mus: Sequence[float], best_mu: float) -> FloatArray | None:
    """Return one additional decade when ``best_mu`` is on a grid boundary."""
    if best_mu == min(mus):
        return np.logspace(np.log10(best_mu) - 1, np.log10(best_mu), 4)[:-1]
    if best_mu == max(mus):
        return np.logspace(np.log10(best_mu), np.log10(best_mu) + 1, 4)[1:]
    return None


def _select_es_mu(cv_results: Sequence[CVResult], minimum_mu: float) -> float | None:
    """Return the first ES local minimum at or above ``minimum_mu``."""
    results = sorted(cv_results, key=attrgetter('mu'))
    for i, result in enumerate(results[:-1]):
        if result.mu < minimum_mu:
            continue
        if result.estimation_stability < results[i + 1].estimation_stability:
            return result.mu
    return None


def search_mu(
        estimator: NCRF,
        data: RegressionData,
        mus: Sequence[float] | str,
        tol: float,
        n_splits: int,
        n_workers: int,
        use_ES: bool,
) -> tuple[float, List[CVResult]]:
    """Cross-validate over ``mus`` and choose the regularization parameter.

    Builds the search grid (from the data when ``mus == 'auto'``), extends it by a
    decade if the best value lands on a boundary, and optionally refines the choice
    with the estimation-stability criterion. Returns the chosen ``mu`` and all
    :class:`CVResult`.
    """
    logger = logging.getLogger(__name__)
    if mus == 'auto':
        mus = find_mu_range(estimator._new_solver().gradient(data))
    logger.info('Crossvalidation initiated!')
    cv_results = crossvalidate(estimator, data, mus, tol, n_splits, n_workers)
    best_mu = select_best_mu(cv_results, 'cross-fit')
    new_mus = _extend_mu_grid(mus, best_mu)

    if new_mus is not None:
        direction = 'left' if new_mus[-1] < best_mu else 'right'
        logger.info(f'CVmu is {best_mu}: extending range of mu towards {direction}')
        cv_results.extend(crossvalidate(estimator, data, new_mus, tol, n_splits, n_workers))
        best_mu = select_best_mu(cv_results, 'cross-fit')

    mu = best_mu
    if use_ES:
        if mu == max(result.mu for result in cv_results):
            logger.info(f'\nCVmu is {mu}: could not find mu based on estimation stability criterion\nContinuing with cross-validation only.')
        else:
            es_mu = _select_es_mu(cv_results, mu)
            if es_mu is None:
                logger.warning('\nNo ES minima found: could not find mu based on estimation stability criterion.\nContinuing with cross-validation only.')
            else:
                mu = es_mu
    return mu, cv_results


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
