"""End-to-end tests for the public ``fit_ncrf()`` workflow and model outputs."""

# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause)
import pickle

from eelbrain import set_time, set_tmin
import numpy as np
import pytest

from ncrf import ChampLasso, fit_ncrf, NCRFModel
from ncrf.tests.fetch import load

from eelbrain import Categorial, concatenate
from eelbrain.testing import assert_dataobj_equal


@pytest.mark.parametrize(
    'mu, expected',
    [
        (0.1, (0.1,)),
        (1, (1.0,)),
        ([0.1], (0.1,)),
        ((0.1, 0.2), (0.1, 0.2)),
        (np.array([0.1, 0.2]), (0.1, 0.2)),
    ],
)
def test_champ_lasso_candidates(mu, expected):
    candidates = ChampLasso(mu).candidates(None, None)
    assert tuple(candidate.mu for candidate in candidates) == expected


@pytest.mark.parametrize('mu', [[], 'invalid', [0.1, 'invalid']])
def test_champ_lasso_candidates_invalid(mu):
    with pytest.raises((TypeError, ValueError)):
        ChampLasso(mu).candidates(None, None)


def test_champ_lasso_auto_candidates(monkeypatch):
    expected = (ChampLasso(0.1), ChampLasso(0.2))
    monkeypatch.setattr(
        ChampLasso,
        'auto_candidates',
        lambda self, forward, data: expected,
    )

    assert ChampLasso().candidates(None, None) == expected


def test_ncrf():
    meg = load('meg').sub(time=(0, 5))
    stim = load('stim').sub(time=(0, 5))
    fwd = load('fwd_sol')
    emptyroom = load('emptyroom')

    # 1 stimulus
    result = fit_ncrf(meg, stim, fwd, emptyroom, tstop=0.2, normalize='l1', mu=0.0019444, n_iter=3, n_iterc=3,
                      n_iterf=10, do_post_normalization=False)
    # the fitted model is a reusable NCRFModel
    assert isinstance(result.model, NCRFModel)
    # check residual and explained var
    np.testing.assert_allclose(result.scores['explained_variance'], 0.00641890144769941, rtol=0.001)
    np.testing.assert_allclose(result.voxelwise_explained_variance.sum(), 0.004410796436808832, rtol=0.001)
    np.testing.assert_allclose(result.scores['cross_fit'], 178.512, rtol=0.001)
    # check scaling
    stim_baseline = stim.mean()
    np.testing.assert_equal(result.model._design.stim_baseline[0], stim_baseline)
    np.testing.assert_equal(result.model._design.stim_scaling[0], (stim - stim_baseline).abs().mean())
    np.testing.assert_allclose(result.model.h.norm('time').norm('source').norm('space'), 6.601677e-10, rtol=0.001)

    # test persistence
    result_2 = pickle.loads(pickle.dumps(result, pickle.HIGHEST_PROTOCOL))
    assert_dataobj_equal(result_2.model.h, result.model.h)
    assert_dataobj_equal(result_2.model.h_scaled, result.model.h_scaled)
    np.testing.assert_equal(result_2.scores['cross_fit'], result.scores['cross_fit'])
    np.testing.assert_equal(result_2.model.basis_std, result.model.basis_std)
    # the model alone round-trips and reproduces h
    model_2 = pickle.loads(pickle.dumps(result.model, pickle.HIGHEST_PROTOCOL))
    assert_dataobj_equal(model_2.h, result.model.h)

    # test Gaussian basis standard deviation
    solver = ChampLasso(mu=0.0019444, n_iter=1, n_iterc=1, n_iterf=1)
    result = fit_ncrf(
        meg, stim, fwd, emptyroom, tstop=0.2, normalize='l1', solver=solver,
        basis_std=0.050,
    )
    assert result.solver is solver
    assert set(result.scores) == {'explained_variance', 'l2_error', 'cross_fit', 'weighted_l2_error'}
    assert result.model.basis_std == 0.050

    # 2 stimuli, one of them 2-d, normalize='l2'
    diff = stim.diff('time')
    stim2 = concatenate([diff.clip(0), diff.clip(max=0)], Categorial('rep', ['on', 'off']))
    result = fit_ncrf(meg, [stim, stim2], fwd, emptyroom, tstop=[0.2, 0.2], normalize='l2', mu=0.0019444, n_iter=3,
                      n_iterc=3, n_iterf=10, do_post_normalization=False)
    # check scaling
    np.testing.assert_equal(result.model._design.stim_baseline[0], stim.mean())
    np.testing.assert_equal(result.model._design.stim_scaling[0], stim.std())
    np.testing.assert_allclose(result.model.h[0].norm('time').norm('source').norm('space'), 7.0088e-10, rtol=0.001)

    # 2 stimuli, different tstarts (-ve)
    diff = stim.diff('time')
    stim2 = concatenate([diff.clip(0), diff.clip(max=0)], Categorial('rep', ['on', 'off']))
    tstart = [-0.1, 0.1]
    tstop = [0.2, 0.3]
    result = fit_ncrf(meg, [stim, stim2], fwd, emptyroom, tstart=tstart, tstop=tstop, normalize='l2', mu=0.0019444, n_iter=3,
                      n_iterc=3, n_iterf=10, do_post_normalization=False)

    # check residual and explained var
    np.testing.assert_allclose(result.scores['explained_variance'], 0.021442823238037034, rtol=0.001)
    np.testing.assert_allclose(result.scores['cross_fit'], 177.15021740565106, rtol=0.001)
    # check start and stop
    np.testing.assert_equal(result.model.tstart, tstart)
    np.testing.assert_equal(result.model.tstop, tstop)
    # check scaling
    np.testing.assert_equal(result.model._design.stim_baseline[0], stim.mean())
    np.testing.assert_equal(result.model._design.stim_scaling[0], stim.std())
    np.testing.assert_allclose(result.model.h[0].norm('time').norm('source').norm('space'), 6.6065539e-10, rtol=0.001)

    # cross-validation
    result = fit_ncrf(meg, stim, fwd, emptyroom, tstop=0.2, normalize='l1', mu='auto', n_iter=1, n_iterc=2, n_iterf=2, do_post_normalization=False)
    np.testing.assert_allclose(result.solver.mu, 0.0203, rtol=0.001)
    result.cv_info()

    # test without multiprocessing
    result_no_mp = fit_ncrf(meg, stim, fwd, emptyroom, tstop=0.2, normalize='l1', mu='auto', n_iter=1, n_iterc=2, n_iterf=2, n_workers=0, do_post_normalization=False)
    assert_dataobj_equal(result_no_mp.model.h, result.model.h)


def test_ncrf_fit_history():
    meg = load('meg').sub(time=(0, 5))
    stim = load('stim').sub(time=(0, 5))
    fwd = load('fwd_sol')
    emptyroom = load('emptyroom')

    fit_kwargs = dict(tstop=0.2, normalize='l1', mu=0.0019444, n_iter=3, n_iterc=2,
                      n_iterf=5, do_post_normalization=False)

    # default: objective/residual accumulate, trajectories are not stored
    result = fit_ncrf(meg, stim, fwd, emptyroom, **fit_kwargs)
    assert len(result.history.objective) >= 1
    assert len(result.history.residual) >= 1
    assert result.history.theta == []
    assert result.history.gamma == []
    assert result.history.sigma_b == []

    # opt-in trajectory storage
    result = fit_ncrf(meg, stim, fwd, emptyroom, store_theta=True, store_gamma=True,
                      store_sigma_b=True, **fit_kwargs)
    assert len(result.history.theta) >= 1
    assert len(result.history.gamma) == len(result.history.theta)
    assert len(result.history.sigma_b) == len(result.history.theta)
    assert all(theta.shape == result.model.theta.shape for theta in result.history.theta)


def test_ncrf_shifted_nonzero_lags():
    meg = load('meg').sub(case=0)
    stim = load('stim').sub(case=0)
    fwd = load('fwd_sol')
    emptyroom = load('emptyroom')

    # Use a short interior excerpt so the shifted windows below can extend by
    # ``shift`` without running past the available test data.
    tstep = meg.get_dim('time').tstep
    segment_start = 20 * tstep
    shift = 5 * tstep
    lag_stop = 10 * tstep
    segment_stop = segment_start + 75 * tstep

    # Baseline fit: predictor and MEG share the same 75-sample window, and the
    # model estimates lags from 0 to ``lag_stop``.
    stim_segment = set_tmin(stim.sub(time=(segment_start, segment_stop)), 0)
    meg_0 = set_time(set_tmin(meg, -segment_start), stim_segment)

    # Positive lag fit: include ``shift`` extra predictor samples at the end.
    # After dropping the first ``shift`` invalid design rows, this uses the same
    # predictor and MEG samples as the baseline fit.
    stim_positive = set_tmin(stim.sub(time=(segment_start, segment_stop + shift)), 0)
    meg_positive = set_time(set_tmin(meg, shift - segment_start), stim_positive)

    # Negative lag fit: keep the predictor window fixed and shift the MEG labels.
    # After dropping the last ``shift`` invalid design rows, the remaining rows
    # use the same predictor and MEG samples as the baseline fit.
    meg_negative = set_time(set_tmin(meg, -shift - segment_start), stim_segment)

    # Guard the test setup itself: all three fits should differ only in the
    # time labels/lag windows, not in the effective numeric samples.
    lag_samples = int(round(lag_stop / tstep))
    shift_samples = int(round(shift / tstep))
    np.testing.assert_array_equal(stim_positive.x[:-shift_samples], stim_segment.x)
    np.testing.assert_array_equal(meg_positive.x[:, lag_samples + shift_samples:], meg_0.x[:, lag_samples:])
    np.testing.assert_array_equal(meg_negative.x[:, shift_samples:-shift_samples], meg_0.x[:, lag_samples:])

    fit_kwargs = dict(
        mu=0.001,
        tol=0,
        verbose=False,
        n_iter=1,
        n_iterc=1,
        n_iterf=5,
        do_post_normalization=False,
    )
    result_0 = fit_ncrf(
        meg_0, stim_segment, fwd, emptyroom, tstart=0, tstop=lag_stop, **fit_kwargs,
    )
    result_positive = fit_ncrf(
        meg_positive, stim_positive, fwd, emptyroom, tstart=shift,
        tstop=lag_stop + shift, **fit_kwargs,
    )
    result_negative = fit_ncrf(
        meg_negative, stim_segment, fwd, emptyroom, tstart=-shift,
        tstop=lag_stop - shift, **fit_kwargs,
    )

    assert np.linalg.norm(result_0.model.theta) > 0
    for result_shifted in (result_positive, result_negative):
        np.testing.assert_allclose(result_shifted.model.theta, result_0.model.theta)
        np.testing.assert_allclose(result_shifted.scores['cross_fit'], result_0.scores['cross_fit'])
        np.testing.assert_allclose(result_shifted.scores['explained_variance'], result_0.scores['explained_variance'])
