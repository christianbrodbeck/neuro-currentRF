"""Tests for low-level stimulus-to-covariate preparation in the NCRF stack."""

# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause)
import numpy as np
import pytest

from ncrf._data import covariate_from_stim
from ncrf._linalg import gaussian_basis
from ncrf._model import _normalize_mu
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
