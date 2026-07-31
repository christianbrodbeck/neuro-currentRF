"""Tests for low-level stimulus-to-covariate preparation in the NCRF stack."""

from dataclasses import dataclass, replace

# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause)
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from ncrf import CrossValidation, SolverFit
from ncrf._crossvalidation import CVResult
from ncrf._data import RegressionData, covariate_from_stim
from ncrf._forward import ForwardModel
from ncrf._linalg import gaussian_basis
from ncrf._model import NCRFEstimator, NCRF
from ncrf._solvers import Solver
from .fetch import load

from eelbrain import Categorial, NDVar, Scalar, Sensor, UTS, concatenate


SENSOR = Sensor([[1., 0, 0], [0, 1, 0], [0, 0, 1]], ['a', 'b', 'c'])


def test_fit_model():
    estimator = NCRFEstimator.__new__(NCRFEstimator)
    estimator.forward = object()
    solver = Mock()
    solver_fit = SolverFit(np.empty((2, 3)))
    solver.solve.return_value = solver_fit
    data = Mock(design=object())

    model, returned_fit = estimator._fit_model(data, solver, True)

    assert model.forward is estimator.forward
    assert model.theta is solver_fit.theta
    assert model.design is data.design
    assert returned_fit is solver_fit
    solver.solve.assert_called_once_with(estimator.forward, data, verbose=True)


@dataclass(frozen=True)
class _ZeroSolver(Solver):
    def solve(self, forward, data, *, verbose=False):
        return SolverFit(np.zeros((1, 1)))


def test_fit_accepts_generic_solver(monkeypatch):
    estimator = NCRFEstimator.__new__(NCRFEstimator)
    data = MagicMock()
    estimator.forward = Mock(
        # unsafe: Mock rejects attributes named assert_* unless told otherwise
        unsafe=True,
        whiten=Mock(return_value=data),
        whitened_lead_field=np.ones((1, 1)),
        lead_field=np.ones((1, 1)),
        lead_field_scaling=1.0,
    )
    data.design = Mock()
    data.covariates = [np.ones((4, 1))]
    data.norm_factor = 1.0
    data.__iter__.side_effect = lambda: iter([
        (np.arange(4, dtype=float)[None, :], np.ones((4, 1))),
    ])
    data.__len__.return_value = 1
    solver = _ZeroSolver()

    result = estimator.fit(data, solver)

    assert result.solver is solver
    assert result.solver_fit.theta.shape == (1, 1)
    assert result.scores == {
        'explained_variance': pytest.approx(0),
        'l2_error': pytest.approx(7),
    }
    prediction = result.model.predict(data)
    np.testing.assert_array_equal(prediction[0], np.zeros((1, 4)))

    candidates = (Mock(), Mock())
    grid_solver = Mock(spec=Solver)
    grid_solver.candidates.return_value = candidates
    selected_solver = _ZeroSolver()
    cv_results = [Mock()]
    select = Mock(return_value=(selected_solver, cv_results))
    monkeypatch.setattr('ncrf._model.select_solver', select)
    cv = CrossValidation(n_splits=4, n_workers=0)

    result = estimator.fit(data, grid_solver, cv=cv)

    select.assert_called_once_with(estimator, data, candidates, cv)
    assert result.solver is selected_solver
    assert result._cv_results is cv_results


def test_default_selection_contract():
    """A solver implementing only solve() gets working selection defaults."""
    solver = _ZeroSolver()
    worse = _ZeroSolver()
    better = _ZeroSolver()
    cv_results = [
        CVResult(worse, {'l2_error': 3.0, 'explained_variance': 0.1, 'estimation_stability': 1.0}),
        CVResult(better, {'l2_error': 1.0, 'explained_variance': 0.4, 'estimation_stability': 2.0}),
    ]

    assert solver.criterion == 'l2_error'
    assert solver.without_history() is solver
    assert solver.candidates(None, None) == (solver,)
    # no extra passes, and the generic criterion picks the smallest l2_error
    assert solver.refine(cv_results) == ()
    assert solver.select(cv_results, CrossValidation()) is better
    # a generic table renders from whatever score keys are present
    assert 'l2_error' in str(solver.cv_table(cv_results, better))


def test_solver_fit_score_defaults_empty():
    """Solvers without their own scores contribute nothing to the score dict."""
    assert SolverFit(np.empty((2, 3))).score(Mock(), Mock()) == {}


def test_whitening_guard():
    data = RegressionData.__new__(RegressionData)
    data.is_whitened = True
    data.sensor_dim = SENSOR
    whitening_filter = object()

    with pytest.raises(ValueError, match="pass accept_whitening=True"):
        data.whiten(whitening_filter)
    assert data.whiten(whitening_filter, accept_whitening=True) is data

    forward = _forward()
    with pytest.raises(ValueError, match="pass accept_whitening=True"):
        forward.whiten(data)
    assert forward.whiten(data, accept_whitening=True) is data


def _synthetic_data(
        scale: str | None = None,
        seed: int = 0,
        tstop: float = 0.05,
        names: tuple[str, str] = ('loud', 'quiet'),
) -> RegressionData:
    """Two-predictor dataset on strongly mismatched stimulus scales."""
    rng = np.random.RandomState(seed)
    time = UTS(0, 0.01, 200)
    meg = [NDVar(rng.normal(size=(3, 200)), (SENSOR, time))]
    stim = [[
        NDVar(rng.normal(size=200) * 100 + 20, (time,), name=names[0]),
        NDVar(rng.normal(size=200) * 0.01 + 0.5, (time,), name=names[1]),
    ]]
    return RegressionData.from_data(meg, stim, 0, tstop, scale=scale)


def _forward(seed: int = 1) -> ForwardModel:
    """Forward model for the sensors of :func:`_synthetic_data`, with 4 sources."""
    rng = np.random.RandomState(seed)
    return ForwardModel(rng.normal(size=(3, 4)), np.eye(3), Scalar('source', range(4)), SENSOR, None)


def _model(design, n_coefficients: int, seed: int = 1) -> NCRF:
    rng = np.random.RandomState(seed)
    return NCRF(_forward(seed), rng.normal(size=(4, n_coefficients)), design)


@pytest.mark.parametrize('scale', ['l1', 'l2', 'spectral'])
def test_normalize_matches_from_data(scale):
    """Applying normalization to covariates == applying it to the stimulus."""
    expected = _synthetic_data(scale)
    raw = _synthetic_data()
    assert not np.allclose(raw.covariates[0], expected.covariates[0])

    normalized = raw.normalize(expected.design)
    np.testing.assert_allclose(normalized.covariates[0], expected.covariates[0])
    # the source dataset is unchanged
    np.testing.assert_allclose(raw.covariates[0], _synthetic_data().covariates[0])


def test_normalize_does_not_write_through_shared_covariates():
    """Datasets sharing covariate arrays must not be normalized behind each other's back."""
    data = _synthetic_data()
    design = _synthetic_data('l2').design
    before = data.covariates[0].copy()

    whitened = data.whiten(np.eye(3))
    assert whitened.covariates[0] is data.covariates[0]  # whitening only copies meg
    normalized = whitened.normalize(design)

    np.testing.assert_allclose(normalized.covariates[0], _synthetic_data('l2').covariates[0])
    # the dataset the whitened view was derived from is untouched
    np.testing.assert_array_equal(data.covariates[0], before)
    assert data.design.stim_scaling is None


def test_normalize_rejects_renormalization():
    data = _synthetic_data('l2')
    other = _synthetic_data('l2', seed=2)

    with pytest.raises(ValueError, match="already carry a different baseline"):
        data.normalize(other.design)
    # applying the same normalization again is a no-op
    np.testing.assert_array_equal(data.normalize(data.design).covariates[0], data.covariates[0])


def test_normalize_rejects_incompatible_design():
    data = _synthetic_data()

    with pytest.raises(ValueError, match="stim_names"):
        data.normalize(_synthetic_data(names=('quiet', 'loud')).design)
    with pytest.raises(ValueError, match="tstop"):
        data.normalize(_synthetic_data(tstop=0.06).design)


def test_predict_requires_fit_normalization():
    """Coefficients fit on normalized covariates must not be applied to raw ones."""
    normalized = _synthetic_data('l2')
    model = _model(normalized.design, normalized.design.n_coefficients)
    raw = _synthetic_data()

    with pytest.raises(ValueError, match="different centering"):
        model.predict(raw)
    with pytest.raises(ValueError, match="different scaling"):
        model.predict(raw.normalize(replace(normalized.design, stim_scaling=None, scale=None)))
    with pytest.raises(ValueError, match="stim_names"):
        model.predict(_synthetic_data(names=('quiet', 'loud')))

    for expected, actual in zip(model.predict(normalized), model.predict(raw.normalize(model.design))):
        np.testing.assert_allclose(expected, actual)


def test_predict_returns_meg_scale():
    """predict() is in the units of the MEG data; whitened=True gives the fitting space."""
    data = _synthetic_data('l2')
    model = _model(data.design, data.design.n_coefficients)
    forward = model.forward

    predicted = model.predict(data)
    whitened = model.predict(data, whitened=True)

    assert not np.allclose(predicted[0], whitened[0])
    for meg_scale, fit_scale in zip(predicted, whitened):
        # whitening and dividing by sqrt(n_times) is exactly what from_data() applied
        np.testing.assert_allclose(np.dot(forward.whitening_filter, meg_scale) / data.norm_factor, fit_scale)
    # whitening the input does not change the prediction, which only uses covariates
    for expected, actual in zip(predicted, model.predict(forward.whiten(data))):
        np.testing.assert_array_equal(expected, actual)


def test_predict_rejects_mismatched_normalization():
    design = _synthetic_data('l2').design
    model = _model(design, design.n_coefficients)
    other = _synthetic_data('l2', seed=2)

    with pytest.raises(ValueError, match="different centering"):
        model.predict(other)


def test_rejects_sensor_mismatch():
    """Data whose sensors differ from the forward model must not be whitened silently."""
    data = _synthetic_data('l2')
    # same channels in a different order: whitening would apply to the wrong channels
    reordered = replace(data, sensor_dim=Sensor([[0., 0, 1], [0, 1, 0], [1, 0, 0]], ['c', 'b', 'a']))

    model = _model(data.design, data.design.n_coefficients)
    with pytest.raises(ValueError, match="same channels in a different order"):
        model.predict(reordered)
    with pytest.raises(ValueError, match="same channels in a different order"):
        model.evaluate(reordered)

    # a genuinely different channel set names the channels that differ
    renamed = replace(data, sensor_dim=Sensor([[1., 0, 0], [0, 1, 0], [0, 0, 1]], ['a', 'b', 'z']))
    with pytest.raises(ValueError, match=r"only in data: \['z'\]; only in forward model: \['c'\]"):
        model.predict(renamed)

    estimator = NCRFEstimator.__new__(NCRFEstimator)
    estimator.forward = model.forward
    with pytest.raises(ValueError, match="sensors do not match"):
        estimator.fit(reordered, _ZeroSolver())


def test_h_scaled():
    """h_scaled restores the original stimulus scale, whichever scaling was used."""
    for scale in [None, 'l1', 'l2', 'spectral']:
        data = _synthetic_data(scale)
        model = _model(data.design, data.design.n_coefficients)
        h, h_scaled = model.h, model.h_scaled
        if scale is None:
            assert h_scaled is h
            continue
        for i, factor in enumerate(data.design.stim_scaling):
            np.testing.assert_allclose(h_scaled[i].x, h[i].x / factor)


def test_h_scaled_matches_unscaled_fit():
    """h_scaled equals the h of an equivalent model whose covariates were not scaled."""
    scaled = _synthetic_data('l2')
    design = scaled.design
    centered = _synthetic_data().normalize(replace(design, stim_scaling=None, scale=None))
    model = _model(design, design.n_coefficients)
    # coefficients on the unscaled covariates that make the same predictions
    unscaled_model = NCRF(model.forward, model.theta / design.expand(design.stim_scaling), centered.design)

    for expected, actual in zip(model.predict(scaled), unscaled_model.predict(centered)):
        np.testing.assert_allclose(expected, actual)
    # h of the unscaled fit is already in stimulus units
    assert unscaled_model.h_scaled is unscaled_model.h
    for expected, actual in zip(unscaled_model.h, model.h_scaled):
        np.testing.assert_allclose(actual.x, expected.x)


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
