"""The NCRF estimator, reusable fitted model, and fit report.

:class:`NCRFEstimator` owns forward-model preparation, candidate selection, and
fitting. :class:`NCRF` contains the resulting coefficients and prediction API,
independent of the solver that produced them. :class:`NCRFFit` bundles that
model with training scores and solver-specific provenance.
"""
# Authors: Proloy Das <email:proloyd94@gmail.com>
#          Christian Brodbeck <email:brodbecc@mcmaster.ca>
# License: BSD (3-clause)
from __future__ import annotations

from functools import cached_property
from collections.abc import Sequence

from eelbrain import NDVar, UTS, fmtxt
import numpy as np

from ._crossvalidation import CrossValidation, CVResult
from ._data import RegressionData
from ._trf_design import TRFDesign
from ._forward import ForwardModel
from ._metrics import Metric, explained_variance, l2_error, merge_scores
from ._repr import _count_repr, _forward_summary
from ._solvers import Solver, SolverFit
from ._typing import FloatArray


class NCRF:
    """Fitted NCRF model that can be applied to compatible datasets.

    Holds the estimated weights together with the forward model and stimulus
    design needed to reconstruct response functions, predict whitened sensor
    data, and evaluate predictions. Reusable and picklable; produced by
    :meth:`NCRFEstimator.fit` and exposed as :attr:`NCRFFit.model`.

    Attributes
    ----------
    forward
        The internal forward-model state (lead field, whitening filter, and
        source, sensor, and orientation dimensions).
    theta
        Fitted NCRF coefficients over the Gaussian basis.
    design
        Stimulus, basis, and normalization metadata of the data the model was fit
        on, including the TRF timing (``design.tstart``, ``design.tstep``,
        ``design.tstop``) and the Gaussian-basis width (``design.basis_std``).
        Pass it to :meth:`~ncrf.RegressionData.normalize` to bring another
        dataset onto the same scale.
    """

    def __init__(
            self,
            forward: ForwardModel,
            theta: FloatArray,
            design: TRFDesign,
    ) -> None:
        self.forward = forward
        self.theta = theta
        self.design = design

    def __repr__(self) -> str:
        n_basis = self.theta.shape[1]
        predictors = tuple(self.design.stim_names)
        return f"<{type(self).__name__}: {_forward_summary(self.forward)}, {_count_repr(n_basis, 'basis coefficient')}, {predictors=}>"

    def _theta_for(self, data: RegressionData) -> FloatArray:
        """Coefficients, after checking that ``data`` is on the scale they were fit on.

        ``theta`` is fit against covariates carrying the normalization recorded in
        :attr:`design`, so any other normalization would silently change the
        predictions.
        """
        design = data.design
        if design is self.design:
            return self.theta
        self.design.assert_compatible(design)
        design.normalization_to(self.design, assert_applied=True)
        return self.theta

    def _predict_whitened(self, theta: FloatArray, covariate: FloatArray) -> FloatArray:
        """Predicted whitened sensor data for one trial's covariate matrix."""
        return np.dot(np.dot(self.forward.whitened_lead_field, theta), covariate.T)

    def predict(
            self,
            data: RegressionData,
            *,
            whitened: bool = False,
    ) -> list[FloatArray]:
        """Predict sensor-space data for each segment.

        Only the covariates are used, so it does not matter whether ``data`` has
        been whitened.

        Parameters
        ----------
        data
            Prepared dataset with a design compatible with the training data, and
            carrying the same normalization (prepare it with ``scale=None`` and
            apply :meth:`~ncrf.RegressionData.normalize` with :attr:`design`).
        whitened
            Predict in the whitened, variance-normalized sensor space the model is
            fit in, i.e. the space :meth:`evaluate` scores in, rather than in the
            units of the original M/EEG data.

        Returns
        -------
        list
            Predicted arrays, one per segment, each shaped
            ``(n_sensors, n_times)``.
        """
        self.forward.assert_sensors(data)
        theta = self._theta_for(data)
        if whitened:
            return [self._predict_whitened(theta, covariate) for covariate in data.covariates]
        # Predicting through the un-whitened lead field, and undoing the sqrt(n_times)
        # by which both MEG and covariates were divided, puts the prediction back into
        # the units of the M/EEG data the dataset was built from.
        source = np.dot(self.forward.lead_field, theta) / self.forward.lead_field_scaling
        return [np.dot(source, covariate.T) * data.norm_factor for covariate in data.covariates]

    def evaluate(
            self,
            data: RegressionData,
            metrics: Sequence[Metric] = (explained_variance, l2_error),
            *,
            accept_whitening: bool = False,
    ) -> dict[str, float]:
        """Score predictions on ``data`` with one or more metrics.

        The data is whitened and predicted once, and every metric is evaluated on
        those predictions. Scoring happens in whitened sensor space, where the noise
        is isotropic and channels are therefore comparable; this is the space the
        solver optimizes in and the one cross-validation compares candidates in. Use
        ``predict(data, whitened=True)`` to obtain the predictions the metrics see.

        Parameters
        ----------
        data
            Prepared dataset with a design compatible with the training data, and
            carrying the same normalization (see :meth:`predict`).
        metrics
            Metric functions such as :func:`~ncrf.explained_variance`, each
            mapping observed and predicted per-segment arrays to a scalar.
            Results are keyed by function name.
        accept_whitening
            Set to ``True`` only when the model's whitening filter was already
            applied to ``data``.
        """
        data = self.forward.whiten(data, accept_whitening)
        theta = self._theta_for(data)
        observed = [meg for meg, _ in data]
        predicted = [self._predict_whitened(theta, covariate) for _, covariate in data]
        return {metric.__name__: metric(observed, predicted) for metric in metrics}

    def voxelwise_explained_variance(
            self,
            data: RegressionData,
            *,
            accept_whitening: bool = False,
    ) -> NDVar:
        """Compute each source's contribution to explained variance.

        ``data`` has to carry the same normalization as the training data (see
        :meth:`predict`). Set ``accept_whitening=True`` only when the model's
        whitening filter was applied to ``data``.
        """
        data = self.forward.whiten(data, accept_whitening)
        theta = self._theta_for(data)
        W_leadfield = self.forward.whitened_lead_field
        temp = np.zeros(len(self.forward.source))
        for meg, covariate in data:
            total_var = np.var(meg, axis=1)
            y_full = meg - self._predict_whitened(theta, covariate)
            base_var = np.var(y_full, axis=1)
            for i in range(len(self.forward.source)):
                # Zeroing source i's weights just removes its (linear) contribution
                # to the prediction, so add that contribution back to the residual.
                block = self.forward.source_block(i)
                contribution = np.dot(np.dot(W_leadfield[:, block], theta[block]), covariate.T)
                y_i = y_full + contribution
                temp[i] += np.nansum((np.var(y_i, axis=1) - base_var) / total_var) / meg.shape[0]

        return NDVar(temp / len(data), self.forward.source)

    @cached_property
    def h(self) -> NDVar | list[NDVar]:
        """The spatio-temporal response function as Eelbrain NDVars.

        Expands the Gabor coefficients in :attr:`theta` back into labeled response
        functions, one per predictor variable (or a bare NDVar when the model was
        fit on a single predictor).

        ``h`` is in source-current units per unit of the *normalized* covariates
        the model was fit on (``scale`` argument). Use :attr:`~ncrf.NCRF.h_scaled`
        for the response in units of the original stimulus.
        """
        design = self.design
        space = self.forward.space
        source_dims = (self.forward.source, space) if space else (self.forward.source,)

        h = []
        start = 0
        for basis, stim_len, dim, name, tstart in zip(design.basis, design.stim_lens, design.stim_dims, design.stim_names, design.tstart):
            # This predictor's columns of theta, as (stim_len, source, basis)
            stop = start + basis.shape[1] * stim_len
            x = self.theta[:, start:stop].reshape((self.theta.shape[0], stim_len, -1))
            x = np.squeeze(x.swapaxes(1, 0))
            start = stop

            x = np.dot(x, basis.T) / self.forward.lead_field_scaling
            time = UTS(tstart, design.tstep, x.shape[-1])
            dims = (dim, *source_dims, time) if dim else (*source_dims, time)
            h.append(NDVar(x.reshape(*(len(d) for d in dims)), dims, name=name))

        if design.stim_is_single:
            return h[0]
        return h

    @cached_property
    def h_scaled(self) -> NDVar | list[NDVar]:
        """:attr:`~ncrf.NCRF.h` with the covariate normalization undone.

        In source-current units per unit of the *original* stimulus, and hence
        comparable across predictors and across models fit with different
        ``scale``. Identical to :attr:`~ncrf.NCRF.h` when the covariates were not
        scaled.
        """
        if self.design.stim_scaling is None:
            return self.h
        scaling = self.design.per_predictor(self.design.stim_scaling)
        if self.design.stim_is_single:
            return self.h / scaling[0]
        return [h / s for h, s in zip(self.h, scaling)]


class NCRFEstimator:
    """Estimator for neuro-current response functions (NCRFs).

    Construct with a lead field and noise covariance, then call :meth:`fit`
    with a :class:`RegressionData` instance to obtain an :class:`NCRFFit`.

    Parameters
    ----------
    lead_field
        Forward solution a.k.a. lead-field matrix, with ``sensor`` and ``source``
        dimensions and an optional ``space`` dimension for free orientation.
    noise_covariance
        Noise covariance matrix in sensor space, typically estimated from empty-room
        recordings.

    Notes
    -----
    Use :meth:`RegressionData.from_data` to prepare the M/EEG and predictor
    segments, initialize this estimator with the matching forward inputs, and
    call :meth:`fit` with a configured :class:`Solver`. The returned
    :class:`NCRFFit` keeps the reusable model separate from solver-specific
    fitted state.
    """

    def __init__(
            self,
            lead_field: NDVar,
            noise_covariance: FloatArray,
    ) -> None:
        self.forward = ForwardModel.from_lead_field(lead_field, noise_covariance)

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {_forward_summary(self.forward)}>'

    def fit_model(
            self,
            data: RegressionData,
            solver: Solver,
            verbose: bool = False,
    ) -> tuple[NCRF, SolverFit]:
        """Fit one fixed solver configuration on prepared, whitened data.

        The single-fit primitive underneath :meth:`fit`: no candidate selection,
        no scoring. Cross-validation uses it to fit the individual folds.

        Parameters
        ----------
        data
            Prepared, whitened data.
        solver
            Fixed solver configuration.
        verbose
            Print intermediate values of the cost functions.
        """
        solver_fit = solver.solve(self.forward, data, verbose=verbose)
        model = NCRF(
            forward=self.forward,
            theta=solver_fit.theta,
            design=data.design,
        )
        return model, solver_fit

    def fit(
            self,
            data: RegressionData,
            solver: Solver,
            *,
            cv: CrossValidation | None = None,
            verbose: bool = False,
            compute_explained_variance: bool = False,
            accept_whitening: bool = False,
    ) -> NCRFFit:
        """Fit a configured solver to prepared regression data.

        Parameters
        ----------
        data
            Prepared M/EEG data and corresponding basis-projected covariates,
            with the same channels in the same order as the lead field. The
            input object is not mutated.
        solver
            Solver configuration. A solver with more than one configuration to
            choose from selects one through cross-validation before the final fit.
        cv
            Cross-validation configuration. Defaults to :class:`CrossValidation`;
            unused when the solver does not cross-validate.
        verbose
            If set True prints intermediate values of the cost functions (default ``False``).
        compute_explained_variance
            Compute the source-wise explained-variance diagnostic and store it
            on the result.
        accept_whitening
            Accept pre-whitened data. This is intended for internal workflows
            that slice an already-whitened dataset, such as cross-validation.

        Returns
        -------
        NCRFFit
            Fitted model, selected solver, solver state, training scores, and
            optional cross-validation and source-wise diagnostics.
        """
        data = self.forward.whiten(data, accept_whitening)
        if cv is None:
            cv = CrossValidation()
        solver, cv_results = solver.search(self, data, cv)
        model, solver_fit = self.fit_model(data, solver, verbose)
        scores = merge_scores(
            model.evaluate(data, accept_whitening=True),
            solver_fit.score(self.forward, data),
        )
        if compute_explained_variance:
            voxelwise = model.voxelwise_explained_variance(data, accept_whitening=True)
        else:
            voxelwise = None

        return NCRFFit(
            model,
            solver=solver,
            solver_fit=solver_fit,
            scores=scores,
            voxelwise_explained_variance=voxelwise,
            cv_results=cv_results or None,
        )


class NCRFFit:
    """Report produced by :meth:`NCRFEstimator.fit`.

    Bundles the fitted :class:`NCRF` with the training-set evaluation and the
    fitting provenance. Model-level predictive quantities live on :attr:`model`;
    solver-specific state lives on :attr:`solver_fit`.

    Attributes
    ----------
    model
        The fitted :class:`NCRF` with TRF reconstruction, prediction, and
        evaluation methods.
    solver
        Immutable solver configuration used for the fit.
    solver_fit
        Solver-specific fitted state and iteration history (for ChampLasso,
        ``solver_fit.history``).
    scores
        Prediction metrics on the training data, keyed by name: the
        solver-independent model metrics plus whatever the solver contributes
        through :meth:`SolverFit.score` (for ChampLasso, ``cross_fit`` and
        ``weighted_l2_error``). For an arbitrary dataset use
        :meth:`NCRF.evaluate`.
    voxelwise_explained_variance
        Source-wise contributions to explained variance on the training data
        (``None`` unless requested at fit time).

    Notes
    -----
    Cross-validation scores are retained when the solver's search cross-validates,
    and are exposed through :meth:`cv_info` and :meth:`cv_mu`.
    """

    def __init__(
            self,
            model: NCRF,
            *,
            solver: Solver,
            solver_fit: SolverFit,
            scores: dict[str, float],
            voxelwise_explained_variance: NDVar | None,
            cv_results: list[CVResult] | None,
    ) -> None:
        self.model = model
        self.solver = solver
        self.solver_fit = solver_fit
        self.scores = scores
        self.voxelwise_explained_variance = voxelwise_explained_variance
        self._cv_results = cv_results

    def __repr__(self) -> str:
        forward = self.model.forward
        solver = type(self.solver).__name__
        scores = self.scores
        voxelwise = self.voxelwise_explained_variance is not None
        return f'<{type(self).__name__}: {_forward_summary(forward)}, solver={solver}, {scores=}, {voxelwise=}>'

    def cv_info(self) -> fmtxt.Table:
        """Summarize stored cross-validation scores in a table."""
        return self.solver.cv_table(self._require_cv_results(), self.solver)

    def cv_mu(self, criterion: str = 'cross-fit') -> float:
        """Retrieve best mu based on cross-validation (:class:`ChampLasso` only)

        Parameters
        ----------
        criterion
            Criterion for best fit. Possible values:

            - ``'cross-fit'``: The smallest cross-fit value (default)
            - ``'l2'``: The smallest l2 error
            - ``'l2/mu'``: The local minimum in the l2 error with smallest mu
        """
        from ._solvers.champ_lasso import select_by_criterion

        return select_by_criterion(self._require_cv_results(), criterion).mu

    def _require_cv_results(self) -> list[CVResult]:
        if not self._cv_results:
            raise ValueError("No cross-validation results; use a solver that searches over several configurations, such as ChampLasso(mu='auto').")
        return self._cv_results
