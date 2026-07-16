"""Tests for low-level stimulus-to-covariate preparation in the NCRF stack."""

# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause)
from unittest.mock import Mock

import numpy as np
import pytest

from ncrf._data import RegressionData, covariate_from_stim
from ncrf._linalg import gaussian_basis
from ncrf._model import NCRF, NCRFModel, _normalize_mu
from ncrf._solver import FitHistory
from .fetch import load

from eelbrain import Categorial, concatenate


@pytest.mark.parametrize(
    'mu, expected',
    [
        (0.1, 0.1),
        (1, 1.0),
        ([0.1], 0.1),
        ((0.1, 0.2), (0.1, 0.2)),
        (np.array([0.1, 0.2]), (0.1, 0.2)),
        ('auto', 'auto'),
    ],
)
def test_normalize_mu(mu, expected):
    assert _normalize_mu(mu) == expected


@pytest.mark.parametrize('mu', [[], 'invalid', [0.1, 'invalid']])
def test_normalize_mu_invalid(mu):
    with pytest.raises((TypeError, ValueError)):
        _normalize_mu(mu)


def test_fit_model(monkeypatch):
    estimator = NCRF.__new__(NCRF)
    solver = Mock()
    monkeypatch.setattr(estimator, '_new_solver', lambda: solver)
    model = object()
    monkeypatch.setattr(NCRFModel, '_from_solver', lambda solver_, data_: model)
    data = object()
    history = FitHistory()

    result = estimator._fit_model(data, 0.1, 1e-5, history, True)

    assert result is model
    solver.run.assert_called_once_with(data, 0.1, 1e-5, history, True)

    estimator._fit_model(data, 0.2, 1e-4)
    internal_history = solver.run.call_args.args[3]
    assert not internal_history.store_objective
    assert not internal_history.store_residual


def test_whitening_guard():
    data = RegressionData.__new__(RegressionData)
    data.is_whitened = True
    whitening_filter = object()

    with pytest.raises(ValueError, match="pass accept_whitening=True"):
        data.whiten(whitening_filter)
    assert data.whiten(whitening_filter, accept_whitening=True) is data

    model = NCRFModel.__new__(NCRFModel)
    model.forward = Mock(whitening_filter=whitening_filter)
    with pytest.raises(ValueError, match="pass accept_whitening=True"):
        model._whiten(data)
    assert model._whiten(data, accept_whitening=True) is data


def test_gaussian_basis():
    basis = gaussian_basis(5, np.linspace(0, 1, 11), 0.1)
    shifted_basis = gaussian_basis(5, np.linspace(10, 11, 11), 0.1)

    assert basis.shape == (11, 4)
    np.testing.assert_allclose(basis, shifted_basis)


def test_covariate_from_stim():
    stim = load('stim')[0]
    # Test if difference between list of stimuli and concatenated stimuli
    diff = stim.diff('time')

    start = [-20, -20]
    stop = [20, 20]
    filter_lengths = np.subtract(stop, start) + 1
    covariates = covariate_from_stim([stim, diff], filter_lengths, start)

    conc = concatenate([stim, diff.clip(0)], Categorial('rep', ['on', 'off']))
    covariates_conc = covariate_from_stim(conc, filter_lengths, start)

    assert np.array(covariates).shape == np.array(covariates_conc).shape
    np.testing.assert_allclose(np.array(covariates)[0, 0, 0], np.array(covariates_conc)[0, 0, 0], rtol=0.001)

    # Test if shifted covariate array is equal to unshifted
    start = [-20]
    stop = [20]
    filter_lengths = np.subtract(stop, start) + 1
    covariates = covariate_from_stim([stim], filter_lengths, start)

    start = [0]
    stop = [40]
    filter_lengths = np.subtract(stop, start) + 1
    covariates_shift = covariate_from_stim([stim], filter_lengths, start)

    assert covariates[0].shape[0] == len(stim.get_dim('time'))
    np.testing.assert_array_equal(covariates[0][:-20], covariates_shift[0][20:])
