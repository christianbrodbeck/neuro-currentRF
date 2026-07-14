"""Tests for cross-validation orchestration and selection."""

from unittest.mock import Mock

import numpy as np

from ncrf import _crossvalidation as cv


def _cv_result(mu, *, cross_fit=0.0, es=0.0):
    return cv.CVResult(mu, 0.0, es, cross_fit, 0.0)


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
    model.eval_obj.assert_called_once_with(test_data, True)
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
