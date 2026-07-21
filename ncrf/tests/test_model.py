"""Tests for low-level stimulus-to-covariate preparation in the NCRF stack."""

from dataclasses import dataclass

# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause)
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from ncrf import CrossValidation
from ncrf._data import RegressionData, covariate_from_stim
from ncrf._linalg import gaussian_basis
from ncrf._model import NCRF, NCRFModel
from ncrf._solvers import Solver, SolverFit
from .fetch import load

from eelbrain import Categorial, concatenate


def test_fit_model():
    estimator = NCRF.__new__(NCRF)
    estimator.forward = object()
    solver = Mock()
    solver_fit = SolverFit(np.empty((2, 3)))
    solver.solve.return_value = solver_fit
    data = Mock(trf_design=object())

    model, returned_fit = estimator._fit_model(data, solver, True)

    assert model.forward is estimator.forward
    assert model.theta is solver_fit.theta
    assert model._design is data.trf_design
    assert returned_fit is solver_fit
    solver.solve.assert_called_once_with(estimator.forward, data, verbose=True)


@dataclass(frozen=True)
class _ZeroSolver(Solver):
    def solve(self, forward, data, *, verbose=False):
        return SolverFit(np.zeros((1, 1)))


def test_fit_accepts_generic_solver(monkeypatch):
    estimator = NCRF.__new__(NCRF)
    estimator.forward = Mock(
        whitening_filter=object(),
        whitened_lead_field=np.ones((1, 1)),
    )
    data = MagicMock()
    data.whiten.return_value = data
    data.trf_design = object()
    data.__iter__.side_effect = lambda: iter([
        (np.arange(4, dtype=float)[None, :], np.ones((4, 1))),
    ])
    data.__len__.return_value = 1
    solver = _ZeroSolver()

    result = estimator.fit(data, solver)

    assert result.solver is solver
    assert result.solver_fit.theta.shape == (1, 1)
    assert result.residual is None
    assert result.explained_var == pytest.approx(0)
    assert result.scores == {
        'explained_variance': pytest.approx(0),
        'l2_error': pytest.approx(7),
    }
    prediction = result.model.predict(data, accept_whitening=True)
    np.testing.assert_array_equal(prediction[0], np.zeros((1, 4)))

    candidates = (Mock(), Mock())
    grid_solver = Mock(spec=Solver)
    grid_solver.candidates.return_value = candidates
    selected_solver = _ZeroSolver()
    cv_results = [Mock()]
    select = Mock(return_value=(selected_solver, cv_results))
    monkeypatch.setattr('ncrf._model.search_mu', select)
    cv = CrossValidation(n_splits=4, n_workers=0)

    result = estimator.fit(data, grid_solver, cv=cv)

    select.assert_called_once_with(estimator, data, candidates, cv)
    assert result.solver is selected_solver
    assert result._cv_results is cv_results


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
