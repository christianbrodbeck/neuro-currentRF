"""High-level orchestration for preparing inputs and fitting an NCRF model.

This module is the public entrypoint for the library. It normalizes the
different supported input layouts, derives stimulus scaling metadata, prepares
:class:`RegressionData`, and then delegates candidate selection and fitting to
:class:`NCRFEstimator`.
"""
# Authors: Proloy Das <email:proloyd94@gmail.com>
#          Christian Brodbeck <email:brodbecc@mcmaster.ca>
#          Marlies Gillis <email: >
# License: BSD (3-clause)
from __future__ import annotations

import collections
from collections.abc import Sequence
from typing import TypeAlias

from eelbrain import NDVar, Sensor
import mne
import numpy as np

from ._crossvalidation import CrossValidation
from ._data import RegressionData
from ._model import NCRFEstimator, NCRFFit
from ._solvers import ChampLasso, Solver
from ._typing import MuArg, ScaleArg


StimulusInput: TypeAlias = NDVar | Sequence[NDVar]
TrialStimulusInput: TypeAlias = StimulusInput | Sequence[StimulusInput]
MegInput: TypeAlias = NDVar | Sequence[NDVar]


def _handle_noise_channels(
        noise: mne.Covariance | NDVar | np.ndarray,
        sensor_dim: Sensor,
) -> np.ndarray:
    """Return the noise covariance aligned to the MEG sensor order."""
    if isinstance(noise, mne.Covariance):
        chs_noise = set(noise.ch_names)
        chs_data = set(sensor_dim.names)
        missing = sorted(chs_data - chs_noise)
        if missing:
            raise RuntimeError(f"Missing channels in noise covariance: {', '.join(missing)}")

        index = [noise.ch_names.index(ch) for ch in sensor_dim.names]

        if noise['diag']:
            full_cov = np.zeros((len(noise.data), len(noise.data)))
            row, col = np.diag_indices(full_cov.shape[0])
            full_cov[row, col] = noise.data
            noise_cov = full_cov[index, :][:, index]
        else:
            noise_cov = noise.data[index, :][:, index]

    elif isinstance(noise, NDVar):
        er = noise.get_data(('sensor', 'time'))
        noise_cov = np.dot(er, er.T) / er.shape[1]

    elif isinstance(noise, np.ndarray):
        n = len(sensor_dim)
        if noise.shape != (n, n):
            raise ValueError(f"noise = array of shape {noise.shape}; should be {(n, n)}")
        noise_cov = noise

    else:
        raise TypeError(f"Invalid noise type: {type(noise)}. Must be NDVar, mne.Covariance, or ndarray.")

    return noise_cov


def fit_ncrf(
        meg: MegInput,
        stim: TrialStimulusInput,
        lead_field: NDVar,
        noise: mne.Covariance | NDVar | np.ndarray,
        tstart: float | Sequence[float] = 0,
        tstop: float | Sequence[float] = 0.5,
        nlevels: int = 1,
        n_iter: int = 10,
        n_iterc: int = 10,
        n_iterf: int = 100,
        scale: ScaleArg = 'spectral',
        in_place: bool = False,
        mu: MuArg = 'auto',
        tol: float = 1e-3,
        verbose: bool = False,
        n_splits: int = 3,
        n_workers: int | None = None,
        use_ES: bool = False,
        basis_std: float = 0.0085,
        solver: Solver | None = None,
) -> NCRFFit:
    r"""One shot function for cortical TRF localization.

    Estimate both TRFs and source variance from the observed MEG data by solving
    the Bayesian optimization problem mentioned in :cite:p:`das2020neuro`.

    Parameters
    ----------
    meg
        Observed data. A single contiguous segment can be passed as an
        :class:`eelbrain.NDVar` with ``sensor`` and ``time`` dimensions, equal-length
        trials can be packed into an NDVar with a ``case`` dimension, and unequal-length
        segments can be supplied as a sequence (e.g., :class:`list` of
        :class:`eelbrain.NDVar`).
    stim
        One or more predictors corresponding to each item in ``meg``. Predictors can
        be supplied as one NDVar per segment or as nested sequences when each segment
        has multiple predictors. Individual predictors may have only a ``time`` axis or
        one additional feature dimension before ``time``.
    lead_field
        Forward solution a.k.a. lead-field matrix.
    noise
        Empty-room noise covariance, either directly as :class:`mne.Covariance`, as an
        :class:`eelbrain.NDVar` from which a covariance will be estimated, or as an
        already aligned covariance matrix. Covariance inputs are checked against the MEG
        sensor order and raise an error when sensors are missing or the shape is wrong.
    tstart
        Start of the TRF in seconds. A scalar applies to all predictors; a sequence
        specifies one start time per predictor.
    tstop
        Stop of the TRF in seconds. A scalar applies to all predictors; a sequence
        specifies one stop time per predictor.
    nlevels
        Decides the density of Gabor atoms. Bigger nlevel -> less dense basis.
        By default it is set to ``1``. ``nlevel > 2`` should be used with caution.
    n_iter
        Number of outer iterations of the algorithm, by default set to 10.
    n_iterc
        Number of Champagne iterations within each outer iteration, by default set to 10.
    n_iterf
        Number of FASTA iterations within each outer iteration, by default set to 100.
    scale
        Normalization applied before model fitting: each predictor's mean is
        subtracted, and the covariates are divided by the ``'l2'`` (standard
        deviation of ``stim``), ``'l1'`` (mean absolute deviation of ``stim``) or
        ``'spectral'`` (average spectral norm of the covariates, which equalizes
        covariate scales across predictor variables; the default) scale. Use
        ``None`` to leave ``stim`` untouched. :attr:`~ncrf.NCRF.h_scaled` undoes the
        scaling, whichever one is used.
    in_place
        By default, ``meg`` is copied to make it independent of the object supplied
        to the function. Set to ``True`` to skip the copy and modify it in place,
        saving memory when working with large datasets. ``stim`` is never modified.
    mu
        Choice of regularizer parameters. Pass a single value to fit one model, a
        sequence to cross-validate over an explicit grid, or ``'auto'`` to derive a
        search range from the data.
    tol
        Tolerance factor deciding stopping criterion for the overall algorithm. The iterations
        are stooped when ``norm(trf_new - trf_old)/norm(trf_old) < tol`` condition is met.
        By default ``tol=1e-3``.
    verbose
        If True prints intermediate results, by default False.
    n_splits
        Number of cross-validation folds. By default it uses 3-fold cross-validation.
    n_workers
        Number of worker processes for cross-validation. If ``None``, use the
        library's configured default.
    use_ES
        Use estimation stability criterion :cite:`limEstimationStabilityCrossValidation2016` to
        choose the best ``mu``. (``False`` by default, see :class:`~ncrf.ChampLasso`).
    basis_std
        Standard deviation of the Gaussian basis atoms in seconds
        (default ``0.0085``, approximately 20 ms FWHM).
        The standard deviation (std) is related to the fwmh by:
        :math:`std = fwhm / (2 * (sqrt(2 * log(2))))`.
    solver
        Solver configuration. When supplied, ``mu``, ``use_ES`` and the iteration
        arguments are ignored; configure them on the solver. ``n_splits`` and
        ``n_workers`` still configure the folds a searching solver is scored on.

    Returns
    -------
    :class:`NCRFFit`
        Fit report. The fitted, reusable model is :attr:`NCRFFit.model` (an
        :class:`NCRF`); the response functions are ``result.model.h`` /
        ``result.model.h_scaled``, and metrics for an arbitrary dataset are
        ``result.model.evaluate(data)``. Training-set metrics (``scores``,
        ``voxelwise_explained_variance``), solver state
        (``solver_fit``, including ``solver_fit.history``), and ``cv_info()``
        live on the result itself.

    Examples
    --------
    MEG data ``y`` with dimensions (case, sensor, time) and predictor ``x``
    with dimensions (case, time)::

        fit_ncrf(y, x, fwd, cov)

    ``x`` can always also have an additional predictor dimension, for example,
    if ``x`` represents a spectrogram: (case, frequency, time). The case
    dimension is optional, i.e. a single contiguous data segment is also
    accepted, but the case dimension should always match between ``y`` and
    ``x``.

    Multiple distinct predictor variables can be supplied as list; e.g., when
    modeling simultaneous responses to an attended and an unattended stimulus
    with ``x_attended`` and ``x_unattended``::

        fit_ncrf(y, [x_attended, x_unattended], fwd, cov)

    Multiple data segments can also be specified as list. E.g., if ``y1`` and
    ``y2`` are responses to stimuli ``x1`` and ``x2``, respoectively::

        fit_ncrf([y1, y2], [x1, x2], fwd, cov)

    And with multiple predictors::

        fit_ncrf([y1, y2], [[x1_attended, x1_unattended], [x2_attended, x2_unattended]], fwd, cov)

    For complete workflows, see the gallery examples
    :ref:`Volume Source Example <sphx_glr_auto_examples_00_example-vol-src.py>`
    and
    :ref:`Surface source Example <sphx_glr_auto_examples_01_example-surf-src.py>`.

    References
    ----------
    .. bibliography::
        :cited:
    """
    # make meg/stim representation uniform:
    meg_trials = []  # [trial_1, trial_2, ...]
    stim_trials = []  # [[trial_1_stim_1, trial_1_stim_2, ...], ...]
    if isinstance(meg, NDVar):
        meg_list = [meg]
        stim_list = [stim]
    elif isinstance(meg, collections.abc.Sequence):
        if len(stim) != len(meg):
            raise ValueError(f"{meg=}, {stim=}: different length")
        meg_list = list(meg)
        stim_list = list(stim)
    else:
        raise TypeError(f"meg={meg!r}")
    stim_is_single = None
    for meg_chunk, stim_chunk in zip(meg_list, stim_list):
        if meg_chunk.has_case:
            n_trials = len(meg_chunk)
            meg_trials.extend(meg_chunk)
        else:
            n_trials = 0
            meg_trials.append(meg_chunk)

        if stim_is_single is None:
            stim_is_single = isinstance(stim_chunk, NDVar)
        elif stim_is_single != isinstance(stim_chunk, NDVar):
            raise ValueError(f"{stim=}: inconsistent element types (NDVar/list)")

        if stim_is_single:
            stim_chunk = [stim_chunk]

        if n_trials:
            if not all(s.has_case and len(s) == n_trials for s in stim_chunk):
                raise ValueError(f"{meg=}, {stim=}: inconsistent number of cases")
            stim_trials.extend(zip(*stim_chunk))
        else:
            if any(s.has_case for s in stim_chunk):
                raise ValueError(f"{meg=}, {stim=}: inconsistent case dimensions")
            stim_trials.append(stim_chunk)

    ds = RegressionData.from_data(
        meg_trials, stim_trials, tstart, tstop, nlevels,
        scale, stim_is_single, basis_std=basis_std, in_place=in_place,
    )

    # noise covariance
    noise_cov = _handle_noise_channels(noise, ds.sensor_dim)

    if lead_field.get_dim('sensor') != ds.sensor_dim:
        lead_field = lead_field.sub(sensor=ds.sensor_dim)

    estimator = NCRFEstimator(lead_field, noise_cov)
    if solver is None:
        solver = ChampLasso(mu=mu, n_iter=n_iter, n_iterc=n_iterc, n_iterf=n_iterf, tol=tol, use_es=use_ES)

    return estimator.fit(
        ds,
        solver,
        cv=CrossValidation(n_splits, n_workers),
        verbose=verbose,
        compute_explained_variance=True,
    )
