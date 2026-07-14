"""Tests for cross-validation orchestration and selection."""

from unittest.mock import Mock

import numpy as np

from ncrf import _crossvalidation as cv


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
