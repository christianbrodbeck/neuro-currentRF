"""Tests for cross-validation orchestration and selection."""

from unittest.mock import Mock

import numpy as np
import pytest

from ncrf import ChampLasso
from ncrf import _crossvalidation as cv
from ncrf._solvers import champ_lasso


def _cv_result(mu, *, cross_fit=0.0, es=0.0, l2_error=0.0):
    return cv.CVResult(ChampLasso(mu=mu), {
        'cross_fit': cross_fit,
        'estimation_stability': es,
        'l2_error': l2_error,
        'weighted_l2_error': 0.0,
    })


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
    monkeypatch.setattr(cv, 'compute_es_metric', lambda models, full_data: 4.0)

    model = Mock()
    model.evaluate.return_value = {'l2_error': 3.0, 'explained_variance': 0.5}
    solver_fit = Mock()
    solver_fit.score.return_value = {'cross_fit': 1.0, 'weighted_l2_error': 2.0}
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
    model.evaluate.assert_called_once_with(test_data, accept_whitening=True)
    solver_fit.score.assert_called_once_with(estimator.forward, test_data)
    assert result.solver is solver
    assert result.scores == {
        'cross_fit': 1.0,
        'weighted_l2_error': 2.0,
        'l2_error': 3.0,
        'explained_variance': 0.5,
        'estimation_stability': 4.0,
    }


def test_refine_mu_grid():
    candidates = tuple(ChampLasso(mu=mu) for mu in (0.1, 0.2, 0.3))

    left = candidates[0].refine(candidates, candidates[0])
    right = candidates[0].refine(candidates, candidates[-1])

    np.testing.assert_allclose([solver.mu for solver in left], np.logspace(-2, -1, 4)[:-1])
    np.testing.assert_allclose([solver.mu for solver in right], np.logspace(np.log10(0.3), np.log10(3), 4)[1:])
    assert candidates[0].refine(candidates, candidates[1]) == ()


def test_select_es_solver():
    results = [
        _cv_result(0.1, es=5.0),
        _cv_result(0.2, es=4.0),
        _cv_result(0.3, es=2.0),
        _cv_result(0.4, es=3.0),
    ]

    assert champ_lasso._select_es_solver(results, 0.2).mu == 0.3
    assert champ_lasso._select_es_solver(results, 0.4) is None


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


def test_select_solver_es_is_independent_of_result_order(monkeypatch):
    candidates = tuple(ChampLasso(mu=mu) for mu in (0.1, 0.2, 0.3, 0.4))
    results = [
        _cv_result(0.1, es=5.0, cross_fit=3.0),
        _cv_result(0.3, es=2.0, cross_fit=2.0),
        _cv_result(0.4, es=3.0, cross_fit=4.0),
        _cv_result(0.2, es=4.0, cross_fit=1.0),
    ]
    monkeypatch.setattr(cv, 'crossvalidate', lambda *args, **kwargs: results.copy())

    solver, returned_results = cv.select_solver(
        object(),
        object(),
        candidates,
        cv.CrossValidation(n_splits=2, n_workers=0, use_es=True),
    )

    assert solver.mu == candidates[2].mu
    assert returned_results == results
