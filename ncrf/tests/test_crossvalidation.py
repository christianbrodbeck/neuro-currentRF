"""Tests for cross-validation orchestration and selection."""

from unittest.mock import Mock

import numpy as np
import pytest

from ncrf import ChampLasso
from ncrf import _crossvalidation as cv


def _cv_result(mu, *, cross_fit=0.0, es=0.0):
    return cv.CVResult(ChampLasso(mu=mu), 0.0, es, cross_fit, 0.0)


class _Progress:
    def __init__(self):
        self.updates = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True

    def update(self, n=1):
        self.updates.append(n)


class _InlinePool:
    def __init__(self, *, processes, initializer, initargs):
        self.initializer = initializer
        self.initargs = initargs

    def __enter__(self):
        self.initializer(*self.initargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def imap_unordered(self, function, values):
        return map(function, values)


def test_score_candidate_uses_estimator_fit_primitive(monkeypatch):
    train_data = object()
    test_data = object()
    data = Mock()
    data.basis = [np.empty((2, 3))]
    data.meg = [[np.empty(10)]]
    data.timeslice.side_effect = [train_data, test_data]

    splitter = Mock()
    splitter.split.return_value = [(np.array([0, 1]), np.array([2]))]
    monkeypatch.setattr(cv, 'TimeSeriesSplit', lambda **kwargs: splitter)
    monkeypatch.setattr(
        cv,
        'l2_error',
        lambda model, test, accept_whitening: 3.0,
    )
    monkeypatch.setattr(cv, 'compute_es_metric', lambda models, full_data: 4.0)

    model = Mock()
    solver_fit = Mock()
    solver_fit.evaluate_objective.return_value = (1.0, 2.0)
    estimator = Mock()
    solver = ChampLasso(mu=0.1, tol=1e-5)
    estimator._fit_model.return_value = model, solver_fit

    result = cv._score_candidate(estimator, data, 2, solver)

    fold_solver = estimator._fit_model.call_args.args[1]
    assert fold_solver.mu == 0.1
    assert fold_solver.tol == solver.tol
    assert not fold_solver.store_objective
    assert not fold_solver.store_residual
    assert not fold_solver.store_theta
    assert not fold_solver.store_gamma
    assert not fold_solver.store_sigma_b
    estimator._fit_model.assert_called_once_with(train_data, fold_solver)
    solver_fit.evaluate_objective.assert_called_once_with(
        estimator.forward, test_data, True,
    )
    assert result.solver is solver
    assert result.cross_fit == 1.0
    assert result.weighted_l2_error == 2.0
    assert result.l2_error == 3.0
    assert result.estimation_stability == 4.0


def test_extend_mu_grid():
    candidates = tuple(ChampLasso(mu=mu) for mu in (0.1, 0.2, 0.3))

    left = cv._extend_mu_grid(candidates, candidates[0])
    right = cv._extend_mu_grid(candidates, candidates[-1])

    np.testing.assert_allclose([solver.mu for solver in left], np.logspace(-2, -1, 4)[:-1])
    np.testing.assert_allclose([solver.mu for solver in right], np.logspace(np.log10(0.3), np.log10(3), 4)[1:])
    assert cv._extend_mu_grid(candidates, candidates[1]) == ()


def test_select_es_solver():
    results = [
        _cv_result(0.1, es=5.0),
        _cv_result(0.2, es=4.0),
        _cv_result(0.3, es=2.0),
        _cv_result(0.4, es=3.0),
    ]

    assert cv._select_es_solver(results, 0.2).mu == 0.3
    assert cv._select_es_solver(results, 0.4) is None


def test_crossvalidate_progress(monkeypatch):
    progress = _Progress()
    monkeypatch.setattr(cv, 'tqdm', lambda **kwargs: progress)
    monkeypatch.setattr(
        cv, '_score_candidate',
        lambda estimator, data, n_splits, solver: _cv_result(solver.mu),
    )
    candidates = tuple(ChampLasso(mu=mu) for mu in (0.1, 0.2, 0.3))

    results = cv.crossvalidate(
        object(), object(), candidates, 2, n_workers=0,
    )

    assert [result.solver.mu for result in results] == [0.1, 0.2, 0.3]
    assert progress.updates == [1, 1, 1]
    assert progress.closed


def test_crossvalidate_propagates_worker_error(monkeypatch):
    progress = _Progress()
    monkeypatch.setattr(cv, 'tqdm', lambda **kwargs: progress)
    monkeypatch.setattr(cv, 'Pool', _InlinePool)

    def score_candidate(estimator, data, n_splits, solver):
        if solver.mu == 0.2:
            raise RuntimeError("worker failed")
        return _cv_result(solver.mu)

    monkeypatch.setattr(cv, '_score_candidate', score_candidate)
    candidates = tuple(ChampLasso(mu=mu) for mu in (0.1, 0.2))

    with pytest.raises(RuntimeError, match="worker failed"):
        cv.crossvalidate(
            object(), object(), candidates, 2, n_workers=2,
        )

    assert progress.updates == [1]
    assert progress.closed


def test_search_mu_es_is_independent_of_result_order(monkeypatch):
    candidates = tuple(ChampLasso(mu=mu) for mu in (0.1, 0.2, 0.3, 0.4))
    results = [
        cv.CVResult(candidates[0], 0.0, 5.0, 3.0, 0.0),
        cv.CVResult(candidates[2], 0.0, 2.0, 2.0, 0.0),
        cv.CVResult(candidates[3], 0.0, 3.0, 4.0, 0.0),
        cv.CVResult(candidates[1], 0.0, 4.0, 1.0, 0.0),
    ]
    monkeypatch.setattr(cv, 'crossvalidate', lambda *args, **kwargs: results.copy())

    solver, returned_results = cv.search_param(
        object(),
        object(),
        candidates,
        cv.CrossValidation(n_splits=2, n_workers=0, use_es=True),
    )

    assert solver is candidates[2]
    assert returned_results == results
