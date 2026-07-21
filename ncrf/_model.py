"""The NCRF estimator, its fitted model, and the fit report.

:class:`NCRF` drives the fit (regularization selection and optimization),
:class:`NCRFModel` is the frozen, reusable result that can be applied to new
data, and :class:`NCRFResult` bundles the model with training-set evaluation
and provenance.
"""
# Authors: Proloy Das <email:proloyd94@gmail.com>
#          Christian Brodbeck <email:brodbecc@mcmaster.ca>
# License: BSD (3-clause)
from __future__ import annotations

from functools import cached_property
from typing import Sequence

from eelbrain import NDVar, fmtxt
import numpy as np

from ._crossvalidation import CrossValidation, CVResult, search_param, select_best_solver
from ._data import RegressionData
from ._forward import ForwardModel
from ._reconstruction import TRFDesign
from ._metrics import Metric, explained_variance, l2_error
from ._solvers import Solver, SolverFit
from ._typing import FloatArray


def _orientation_repr(name: str, forward: ForwardModel) -> str:
    """Shared ``repr`` for the estimator/model/result trio."""
    orientation = 'free' if forward.space else 'fixed'
    return f"<[{orientation} orientation] {name} on {forward.source!r}>"


class NCRFModel:
    """Frozen, fitted NCRF model that can be applied to arbitrary datasets.

    Holds the estimated weights together with the forward model and stimulus
    metadata needed to evaluate (and, in the future, predict) on any
    :class:`RegressionData`.  Reusable and picklable; produced by :meth:`NCRF.fit`
    and exposed as :attr:`NCRFResult.model`.

    Attributes
    ----------
    forward
        The :class:`ForwardModel` (lead field, whitening filter, source/sensor/space).
    theta
        NCRF coefficients over the Gabor basis; the frozen weights.
    tstart, tstep, tstop, basis_std
        TRF timing and Gaussian-basis width.
    """
    _name = 'cTRFs estimator'

    def __init__(
            self,
            forward: ForwardModel,
            theta: FloatArray,
            design: TRFDesign,
    ) -> None:
        self.forward = forward
        self.theta = theta
        self._design = design

    @property
    def tstart(self) -> list[float]:
        return self._design.tstart

    @property
    def tstep(self) -> float:
        return self._design.tstep

    @property
    def tstop(self) -> list[float]:
        return self._design.tstop

    @property
    def basis_std(self) -> float:
        return self._design.basis_std

    def __repr__(self) -> str:
        return _orientation_repr(self._name, self.forward)

    def _whiten(
            self,
            data: RegressionData,
            accept_whitening: bool = False,
    ) -> RegressionData:
        """Whiten ``data``, optionally accepting a previously whitened dataset."""
        return data.whiten(self.forward.whitening_filter, accept_whitening=accept_whitening)

    def _predict_whitened(self, covariate: FloatArray) -> FloatArray:
        """Predicted whitened sensor data for one trial's covariate matrix."""
        return np.dot(np.dot(self.forward.whitened_lead_field, self.theta), covariate.T)

    def predict(
            self,
            data: RegressionData,
            *,
            accept_whitening: bool = False,
    ) -> list[FloatArray]:
        """Predict whitened sensor-space data for each segment."""
        data = self._whiten(data, accept_whitening)
        return [self._predict_whitened(covariate) for _, covariate in data]

    def evaluate(
            self,
            data: RegressionData,
            metrics: Sequence[Metric] = (explained_variance, l2_error),
            *,
            accept_whitening: bool = False,
    ) -> dict[str, float]:
        """Score predictions on ``data`` with one or more metrics.

        The data is whitened and predicted once, and every metric is evaluated on
        those predictions.

        Parameters
        ----------
        data
            Dataset to predict and score.
        metrics
            Metric functions from :mod:`ncrf._metrics`, each mapping observed and
            predicted per-segment arrays to a scalar. Results are keyed by
            function name.
        accept_whitening
            Set to ``True`` only when the model's whitening filter was already
            applied to ``data``.
        """
        data = self._whiten(data, accept_whitening)
        observed = [meg for meg, _ in data]
        predicted = [self._predict_whitened(covariate) for _, covariate in data]
        return {metric.__name__: metric(observed, predicted) for metric in metrics}

    def voxelwise_explained_variance(
            self,
            data: RegressionData,
            *,
            accept_whitening: bool = False,
    ) -> NDVar:
        """Compute each source's contribution to explained variance.

        Set ``accept_whitening=True`` only when the model's whitening filter was
        applied to ``data``.
        """
        data = self._whiten(data, accept_whitening)
        W_leadfield = self.forward.whitened_lead_field
        temp = np.zeros(len(self.forward.source))
        for meg, covariate in data:
            total_var = np.var(meg, axis=1)
            y_full = meg - self._predict_whitened(covariate)
            base_var = np.var(y_full, axis=1)
            for i in range(len(self.forward.source)):
                # Zeroing source i's weights just removes its (linear) contribution
                # to the prediction, so add that contribution back to the residual.
                block = self.forward.source_block(i)
                contribution = np.dot(np.dot(W_leadfield[:, block], self.theta[block]), covariate.T)
                y_i = y_full + contribution
                temp[i] += np.nansum((np.var(y_i, axis=1) - base_var) / total_var) / meg.shape[0]

        return NDVar(temp / len(data), self.forward.source)

    @cached_property
    def h_scaled(self) -> NDVar | list[NDVar]:
        """Return ``h`` with the original stimulus scaling restored."""
        return self._design.reconstruct_scaled(self.h)

    @cached_property
    def h(self) -> NDVar | list[NDVar]:
        """Return the spatio-temporal response function as Eelbrain NDVars."""
        return self._design.reconstruct(self.theta, self.forward)


class NCRF:
    """Estimator for neuro-current response functions (cTRFs).

    Construct with a forward model and noise covariance, then call :meth:`fit`
    with a :class:`RegressionData` instance to obtain an :class:`NCRFResult`.

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
    Usage:

    1. Use :meth:`RegressionData.from_data` to construct a prepared dataset
       from MEG and stimulus segments.
    2. Initialize :class:`NCRF` with the lead field and noise covariance.
    3. Call :meth:`NCRF.fit` with the data and a configured :class:`Solver`.
    """
    _name = 'cTRFs estimator'

    def __init__(
            self,
            lead_field: NDVar,
            noise_covariance: FloatArray,
    ) -> None:
        self.forward = ForwardModel.from_lead_field(lead_field, noise_covariance)

    def __repr__(self) -> str:
        return _orientation_repr(self._name, self.forward)

    def _fit_model(
            self,
            data: RegressionData,
            solver: Solver,
            verbose: bool = False,
    ) -> tuple[NCRFModel, SolverFit]:
        """Fit one solver configuration on prepared, whitened data."""
        solver_fit = solver.solve(self.forward, data, verbose=verbose)
        model = NCRFModel(
            forward=self.forward,
            theta=solver_fit.theta,
            design=data.trf_design,
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
    ) -> NCRFResult:
        """Fit a configured solver to prepared regression data.

        Parameters
        ----------
        data
            M/EEG data and the corresponding stimulus variables. Not mutated.
        solver
            Solver configuration. Solvers that expose multiple candidates are
            selected through cross-validation before the final fit.
        cv
            Cross-validation configuration. The default is used when ``solver``
            exposes multiple candidates and ``cv`` is omitted.
        verbose
            If set True prints intermediate values of the cost functions (default ``False``).
        compute_explained_variance
            Compute voxel-wise explained variance.
        accept_whitening
            Accept pre-whitened data. This is intended for internal workflows
            that slice an already-whitened dataset, such as cross-validation.

        Returns
        -------
        NCRFResult
            The fitted model and estimated cortical TRFs.
        """
        data = data.whiten(
            self.forward.whitening_filter,
            accept_whitening=accept_whitening,
        )

        candidates = solver.candidates(self.forward, data)
        if not candidates:
            raise ValueError("solver produced no candidate configurations")
        if len(candidates) == 1:
            solver = candidates[0]
            cv_results = None
        else:
            if cv is None:
                cv = CrossValidation()
            solver, cv_results = search_param(self, data, candidates, cv)

        model, solver_fit = self._fit_model(data, solver, verbose)
        residual = solver_fit.evaluate_objective(self.forward, data)
        scores = model.evaluate(data, accept_whitening=True)
        if compute_explained_variance:
            voxelwise = model.voxelwise_explained_variance(data, accept_whitening=True)
        else:
            voxelwise = None

        return NCRFResult(
            model,
            solver=solver,
            solver_fit=solver_fit,
            scores=scores,
            voxelwise_explained_variance=voxelwise,
            residual=residual,
            cv_results=cv_results,
        )


class NCRFResult:
    """Report produced by :meth:`NCRF.fit`.

    Bundles the fitted :class:`NCRFModel` with the training-set evaluation and the
    fitting provenance. Model-level predictive quantities live on :attr:`model`;
    solver-specific state lives on :attr:`solver_fit`.

    Attributes
    ----------
    model
        The fitted :class:`NCRFModel` (frozen weights + prediction/evaluation API).
    solver
        Immutable solver configuration used for the fit.
    solver_fit
        Solver-specific fitted state and iteration history.
    scores
        Common prediction metrics on the training data, keyed by metric name. For
        an arbitrary dataset use :meth:`model.evaluate`.
    voxelwise_explained_variance
        Source-wise contributions to explained variance on the training data
        (``None`` unless requested at fit time).
    residual
        Solver-specific training objective, or ``None`` when the solver does not
        define one. Retained as a compatibility attribute for ChampLasso.
    history
        Solver-specific per-iteration history, when available.
    """
    _name = 'cTRFs estimator'

    def __init__(
            self,
            model: NCRFModel,
            *,
            solver: Solver,
            solver_fit: SolverFit,
            scores: dict[str, float],
            voxelwise_explained_variance: NDVar | None,
            residual: float | None,
            cv_results: list[CVResult] | None,
    ) -> None:
        self.model = model
        self.solver = solver
        self.solver_fit = solver_fit
        self.scores = scores
        self.voxelwise_explained_variance = voxelwise_explained_variance
        self.residual = residual
        self.history = getattr(solver_fit, 'history', None)
        self._cv_results = cv_results

    def __repr__(self) -> str:
        return _orientation_repr(self._name, self.model.forward)

    def cv_info(self) -> fmtxt.Table:
        """Summarize stored cross-validation scores in a table."""
        if self._cv_results is None:
            raise ValueError(
                "No cross-validation results; use a solver with multiple "
                "candidates, such as ChampLasso(mu='auto').",
            )
        cv_results = sorted(self._cv_results, key=lambda result: result.solver.mu)
        criteria = ('cross-fit', 'l2/mu')
        best_mu = {criterion: self.cv_mu(criterion) for criterion in criteria}

        table = fmtxt.Table('lllll')
        table.cells('mu', 'cross-fit', 'l2-error', 'weighted l2-error', 'ES metric')
        table.midrule()
        fmt = '%.5f'
        for result in cv_results:
            mu = result.solver.mu
            table.cell(fmtxt.stat(mu, fmt=fmt))
            star = 1 if mu == best_mu['cross-fit'] else 0
            table.cell(fmtxt.stat(result.cross_fit, fmt, star, 1))
            star = 1 if mu == best_mu['l2/mu'] else 0
            table.cell(fmtxt.stat(result.l2_error, fmt, star, 1))
            table.cell(fmtxt.stat(result.weighted_l2_error, fmt=fmt))
            table.cell(fmtxt.stat(result.estimation_stability, fmt=fmt))
        # warnings
        mus = [result.solver.mu for result in self._cv_results]
        warnings = []
        if self.solver.mu == min(mus):
            warnings.append("Best mu is smallest mu")
        if warnings:
            table.caption(f"Warnings: {'; '.join(warnings)}")
        return table

    def cv_mu(self, criterion: str = 'cross-fit') -> float:
        """Retrieve best mu based on cross-validation

        Parameters
        ----------
        criterion
            Criterion for best fit. Possible values:

            - ``'cross-fit'``: The smallest cross-fit value (default)
            - ``'l2'``: The smallest l2 error
            - ``'l2/mu'``: The local minimum in the l2 error with smallest mu
        """
        return select_best_solver(self._cv_results, criterion).mu
