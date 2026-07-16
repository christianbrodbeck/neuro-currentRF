"""Tests for cross-validation orchestration and selection."""

from unittest.mock import Mock

import numpy as np
import pytest

from ncrf import _crossvalidation as cv


def _cv_result(mu, *, cross_fit=0.0, es=0.0):
    return cv.CVResult(mu, 0.0, es, cross_fit, 0.0)


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


def test_score_mu_uses_estimator_fit_primitive(monkeypatch):
    train_data = object()
    test_data = object()
    data = Mock()
    data.basis = [np.empty((2, 3))]
    data.meg = [[np.empty(10)]]
    data.timeslice.side_effect = [train_data, test_data]

    splitter = Mock()
    splitter.split.return_value = [(np.array([0, 1]), np.array([2]))]
    monkeypatch.setattr(cv, 'TimeSeriesSplit', lambda **kwargs: splitter)
    monkeypatch.setattr(cv, 'eval_l2', lambda model, test: 3.0)
    monkeypatch.setattr(cv, 'compute_es_metric', lambda models, full_data: 4.0)

    model = Mock()
    model.eval_obj.return_value = (1.0, 2.0)
    estimator = Mock()
    estimator._fit_model.return_value = model

    result = cv._score_mu(estimator, data, n_splits=2, tol=1e-5, mu=0.1)

    estimator._fit_model.assert_called_once_with(train_data, 0.1, 1e-5)
    model.eval_obj.assert_called_once_with(test_data, True, accept_whitening=True)
    assert result.mu == 0.1
    assert result.cross_fit == 1.0
    assert result.weighted_l2_error == 2.0
    assert result.l2_error == 3.0
    assert result.estimation_stability == 4.0


def test_extend_mu_grid():
    mus = (0.1, 0.2, 0.3)

    left = cv._extend_mu_grid(mus, 0.1)
    right = cv._extend_mu_grid(mus, 0.3)

    np.testing.assert_allclose(left, np.logspace(-2, -1, 4)[:-1])
    np.testing.assert_allclose(right, np.logspace(np.log10(0.3), np.log10(3), 4)[1:])
    assert cv._extend_mu_grid(mus, 0.2) is None


def test_select_es_mu():
    results = [
        _cv_result(0.1, es=5.0),
        _cv_result(0.2, es=4.0),
        _cv_result(0.3, es=2.0),
        _cv_result(0.4, es=3.0),
    ]

    assert cv._select_es_mu(results, 0.2) == 0.3
    assert cv._select_es_mu(results, 0.4) is None


def test_crossvalidate_progress(monkeypatch):
    progress = _Progress()
    monkeypatch.setattr(cv, 'tqdm', lambda **kwargs: progress)
    monkeypatch.setattr(
        cv, '_score_mu',
        lambda estimator, data, n_splits, tol, mu: _cv_result(mu),
    )

    results = cv.crossvalidate(object(), object(), (0.1, 0.2, 0.3), 1e-5, 2, n_workers=0)

    assert [result.mu for result in results] == [0.1, 0.2, 0.3]
    assert progress.updates == [1, 1, 1]
    assert progress.closed


def test_crossvalidate_propagates_worker_error(monkeypatch):
    progress = _Progress()
    monkeypatch.setattr(cv, 'tqdm', lambda **kwargs: progress)
    monkeypatch.setattr(cv, 'Pool', _InlinePool)

    def score_mu(estimator, data, n_splits, tol, mu):
        if mu == 0.2:
            raise RuntimeError("worker failed")
        return _cv_result(mu)

    monkeypatch.setattr(cv, '_score_mu', score_mu)

    with pytest.raises(RuntimeError, match="worker failed"):
        cv.crossvalidate(object(), object(), (0.1, 0.2), 1e-5, 2, n_workers=2)

    assert progress.updates == [1]
    assert progress.closed


def test_search_mu_es_is_independent_of_result_order(monkeypatch):
    results = [
        _cv_result(0.1, cross_fit=3.0, es=5.0),
        _cv_result(0.3, cross_fit=2.0, es=2.0),
        _cv_result(0.4, cross_fit=4.0, es=3.0),
        _cv_result(0.2, cross_fit=1.0, es=4.0),
    ]
    monkeypatch.setattr(cv, 'crossvalidate', lambda *args, **kwargs: results.copy())

    mu, returned_results = cv.search_mu(
        object(), object(), (0.1, 0.2, 0.3, 0.4), 1e-5, 2, 0, True,
    )

    assert mu == 0.3
    assert returned_results == results
