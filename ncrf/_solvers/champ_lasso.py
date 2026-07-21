"""Champagne/Lasso NCRF solver.

This module contains the complete high-level implementation of the original
NCRF optimization algorithm. :class:`ChampLasso` is an immutable solver
configuration; :class:`_ChampLassoState` is private mutable state for one run.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field, replace
from math import log10, sqrt
from multiprocessing import current_process
from numbers import Real
from typing import TYPE_CHECKING, Sequence

from eelbrain import fmtxt
import numpy as np
from scipy import linalg
from scipy.signal import find_peaks
from tqdm import tqdm

from .._fastac import Fasta
from .._data import RegressionData
from .._forward import ForwardModel
from .._initialization import mne_initialization
from .._linalg import _inv_sqrtm, compute_gamma
from .._penalties import g, g_group, proxg_group_opt, shrink
from .._typing import _R_tol, FloatArray, GradientFunction, MuArg, ObjectiveFunction
from .base import Solver, SolverFit

if TYPE_CHECKING:
    from .._crossvalidation import CrossValidation, CVResult

#: Per-iteration storage flags, shared by :class:`ChampLasso` and :class:`ChampLassoHistory`.
_STORE_FIELDS = ('store_objective', 'store_residual', 'store_theta', 'store_gamma', 'store_sigma_b')


@dataclass
class ChampLassoHistory:
    """Per-iteration quantities accumulated during fitting.

    Each ``store_*`` flag selects whether the matching quantity is retained.
    :meth:`record` appends to a list only when its flag is set, so the amount of
    stored history can range from nothing to the full optimization trajectory.

    Attributes
    ----------
    objective
        Objective value after each outer iteration.
    residual
        Relative change in ``theta`` after each outer iteration (the convergence
        criterion).
    theta, gamma, sigma_b
        Trajectories of the corresponding solver quantities; populated only when
        the matching ``store_*`` flag is set.
    """
    store_objective: bool = True
    store_residual: bool = True
    store_theta: bool = False
    store_gamma: bool = False
    store_sigma_b: bool = False
    objective: list[float] = field(default_factory=list)
    residual: list[float] = field(default_factory=list)
    theta: list[FloatArray] = field(default_factory=list)
    gamma: list = field(default_factory=list)
    sigma_b: list = field(default_factory=list)

    def record(
            self,
            *,
            objective: float = None,
            residual: float = None,
            theta: FloatArray = None,
            gamma: object = None,
            sigma_b: object = None,
    ) -> None:
        """Append the supplied quantities for which storage is enabled."""
        if self.store_objective and objective is not None:
            self.objective.append(objective)
        if self.store_residual and residual is not None:
            self.residual.append(residual)
        if self.store_theta and theta is not None:
            self.theta.append(theta.copy())
        if self.store_gamma and gamma is not None:
            self.gamma.append(copy.deepcopy(gamma))
        if self.store_sigma_b and sigma_b is not None:
            self.sigma_b.append(copy.deepcopy(sigma_b))


def _evaluate_objective(
        forward: ForwardModel,
        theta: FloatArray,
        Sigma_b: list,
        data: RegressionData,
        return_wl2: bool = False,
) -> float | tuple[float, float]:
    """Evaluate the ChampLasso objective on whitened data."""
    ll2 = 0
    logdet = 0
    for key, (meg, covariate) in enumerate(data):
        y = meg - np.dot(np.dot(forward.whitened_lead_field, theta), covariate.T)
        Cb = np.dot(y, y.T)  # empirical data covariance
        try:
            yhat = linalg.cholesky(Cb, lower=True)
        except np.linalg.LinAlgError:
            hi = y.shape[0] - 1
            lo = max(y.shape[0] - y.shape[1], 0)
            e, v = linalg.eigh(Cb, subset_by_index=(lo, hi))
            tol = e[-1] * _R_tol
            indices = e > tol
            yhat = v[:, indices] * np.sqrt(e[indices])

        sigma_b = Sigma_b[key]
        try:
            Lc = linalg.cholesky(sigma_b, lower=True)
            y = linalg.solve(Lc, yhat)
            logdet_ = np.log(np.diag(Lc)).sum()
        except np.linalg.LinAlgError:
            Lc, e = _inv_sqrtm(sigma_b, return_eig=True)
            y = np.dot(Lc, yhat)
            logdet_ = -np.log(e).sum()

        ll2 += 0.5 * (y ** 2).sum()
        logdet += logdet_
    if return_wl2:
        return (ll2 + logdet) / len(data), ll2 / len(data)
    return (ll2 + logdet) / len(data)


class _ChampLassoState:
    """Mutable state for one :meth:`ChampLasso.solve` call."""

    def __init__(
            self,
            forward: ForwardModel,
            n_iter: int,
            n_iterc: int,
            n_iterf: int,
    ) -> None:
        # configuration (immutable)
        self.forward = forward
        self.n_iter = n_iter
        self.n_iterc = n_iterc
        self.n_iterf = n_iterf
        # initialization seeds (set by _initialize)
        self._init_gamma: list | None = None
        self._init_sigma_b: list[FloatArray] | None = None
        # working estimate (set during run())
        self.theta: FloatArray | None = None
        self.Gamma: list | None = None
        self.Sigma_b: list[FloatArray] | None = None

    def _initialize(self, data: RegressionData) -> None:
        """Seed the working state with a minimum-norm estimate."""
        # MNE-based seeds (re-read by _solve on every Champagne solve)
        self._init_gamma = []
        self._init_sigma_b = []
        for y, _ in data:
            t = y.shape[1]
            gamma, data_cov = mne_initialization(y * (t ** 0.5), self.forward.whitened_lead_field)
            gamma = np.reshape(gamma, (-1, self.forward.dc))
            self._init_gamma.append([np.diag(g) for g in gamma])
            self._init_sigma_b.append(self.forward.whitened_noise_covariance + data_cov)
        # working estimate, seeded from the above
        self.Gamma = [copy.deepcopy(g) for g in self._init_gamma]
        self.Sigma_b = [s.copy() for s in self._init_sigma_b]
        l = sum(basis.shape[1] * (len(dim) if dim else 1) for basis, dim in zip(data.basis, data.stim_dims))
        self.theta = np.zeros((self.forward.lead_field.shape[1], l), dtype=np.float64)

    def _solve(
            self,
            data: RegressionData,
            theta: FloatArray,
            n_iterc: int | None = None,
    ) -> None:
        """Champagne steps implementation

        Parameters
        ----------
        data
            Whitened regression data to fit.
        theta
            Coefficients of the TRFs over the Gabor basis.

        Notes
        -----
        Implementation details can be found at:
        D. P. Wipf, J. P. Owen, H. T. Attias, K. Sekihara, and S. S. Nagarajan,
        “Robust Bayesian estimation of the location, orientation, and time course
        of multiple correlated neural sources using MEG,” NeuroImage, vol. 49,
        no. 1, pp. 641–655, 2010
        """
        logger = logging.getLogger('Champagne')
        # Choose dc
        if self.forward.space:
            dc = len(self.forward.space)
        else:
            dc = 1

        if n_iterc is None:
            n_iterc = self.n_iterc

        logger.debug('Champagne Iterations start:')
        logger.debug('trial \t time taken')
        for key, (meg, covariates) in enumerate(data):
            start = time.time()
            y = meg - np.dot(np.dot(self.forward.whitened_lead_field, theta), covariates.T)
            Cb = np.dot(y, y.T)  # empirical data covariance

            hi = y.shape[0] - 1
            lo = max(y.shape[0] - y.shape[1], 0)
            e, v = linalg.eigh(Cb, subset_by_index=(lo, hi))
            tol = e[-1] * _R_tol
            indices = e > tol
            yhat = v[:, indices] * np.sqrt(e[indices])[None, :]

            gamma = copy.deepcopy(self._init_gamma[key])
            sigma_b = self._init_sigma_b[key].copy()

            # champagne iterations
            for it in range(n_iterc):
                # pre-compute some useful matrices
                try:
                    Lc = linalg.cholesky(sigma_b, lower=True)
                    lhat = linalg.solve(Lc, self.forward.whitened_lead_field)
                    ytilde = linalg.solve(Lc, yhat)
                except np.linalg.LinAlgError:
                    Lc = _inv_sqrtm(sigma_b)
                    lhat = np.dot(Lc, self.forward.whitened_lead_field)
                    ytilde = np.dot(Lc, yhat)

                # compute sigma_b for the next iteration
                sigma_b[:] = self.forward.whitened_noise_covariance[:]

                for i in range(len(self.forward.source)):
                    block = self.forward.source_block(i)
                    if dc > 1:
                        # update Xi
                        x = np.dot(gamma[i], np.dot(lhat[:, block].T, ytilde))
                        # update Zi
                        z = np.dot(lhat[:, block].T, lhat[:, block])
                    else:
                        # update Xi
                        x = gamma[i] * lhat[:, i].T.dot(ytilde)
                        # update Zi
                        z = (lhat[:, i] ** 2).sum()

                    # update Ti
                    gamma[i] = compute_gamma(z, x, dc)

                    # update sigma_b for next iteration
                    lead_block = self.forward.whitened_lead_field[:, block]
                    sigma_b += np.dot(lead_block, np.dot(gamma[i], lead_block.T))

            self.Gamma[key] = gamma
            self.Sigma_b[key] = sigma_b
            end = time.time()
            logger.debug(f'{key} \t {end - start}')

    def run(
            self,
            data: RegressionData,
            mu: float,
            tol: float,
            history: ChampLassoHistory,
            verbose: bool = False,
    ) -> None:
        """Run the alternating FASTA/Champagne optimization for regularization ``mu``.

        Leaves ``theta``, ``Gamma`` and ``Sigma_b`` populated and records the
        requested per-iteration quantities into ``history``.
        """
        logger = logging.getLogger(__name__)
        self._initialize(data)

        if self.forward.space:
            def g_funct(x): return g_group(x, mu)
            def prox_g(x, t): return proxg_group_opt(x, mu * t)
        else:
            def g_funct(x): return g(x, mu)
            def prox_g(x, t): return shrink(x, mu * t)

        theta = self.theta
        myname = current_process().name

        if verbose:
            iter_o = tqdm(range(self.n_iter))
        else:
            iter_o = range(self.n_iter)

        logger.debug('process:iteration \t objective value \t %% change')
        for i in iter_o:
            funct, grad_funct = self._construct_f(data)
            logger.debug(f"Before FASTA:{funct(self.theta)}")
            Theta = Fasta(funct, g_funct, grad_funct, prox_g, n_iter=self.n_iterf)
            Theta.learn(theta)

            residual = self._residual(theta, Theta.coefs_)
            history.record(residual=residual)
            theta = Theta.coefs_
            self.theta = theta
            logger.debug(f"After FASTA: {funct(self.theta)}")

            if residual < tol:
                break

            self._solve(data, theta)
            objective = _evaluate_objective(self.forward, self.theta, self.Sigma_b, data)
            history.record(objective=objective, theta=self.theta, gamma=self.Gamma, sigma_b=self.Sigma_b)
            logger.debug(f'{myname}:{i} \t {objective} \t {residual * 100}')

    def _construct_f(self, data: RegressionData) -> tuple[ObjectiveFunction, GradientFunction]:
        """Build the smooth objective and gradient passed to FASTA.

        Parameters
        ----------
        data
            Prepared regression data.
        """
        leadfields = []
        bEs = []
        bbts = []
        for i in range(len(data)):
            Linv = _inv_sqrtm(self.Sigma_b[i])
            leadfields.append(np.dot(Linv, self.forward.whitened_lead_field))
            bEs.append(np.dot(Linv, data.bE[i]))
            bbts.append(np.trace(np.dot(Linv, np.dot(Linv, data.bbt[i]).T)))

        def f(L, x, bbt, bE, EtE):
            Lx = np.dot(L, x)
            y = bbt - 2 * np.sum(bE * Lx) + np.sum(Lx * np.dot(Lx, EtE))
            return 0.5 * y

        def gradf(L, x, bE, EtE):
            y = bE - np.dot(np.dot(L, x), EtE)
            return -np.dot(L.T, y)

        def funct(x):
            fval = 0.0
            for i in range(len(data)):
                fval = fval + f(leadfields[i], x, bbts[i], bEs[i], data.EtE[i])
            return fval

        def grad_funct(x):
            grad = gradf(leadfields[0], x, bEs[0], data.EtE[0]).astype(np.float64)
            for i in range(1, len(data)):
                grad += gradf(leadfields[i], x, bEs[i], data.EtE[i])
            return grad

        return funct, grad_funct

    def gradient(self, data: RegressionData) -> FloatArray:
        """Per-source gradient magnitude of the data-fit term at the zero estimate.

        Runs an unregularized warm covariance solve and returns the magnitude of
        the smooth objective's gradient at ``theta = 0``, used to calibrate the
        automatic regularization grid. Independent of ``mu``.
        """
        self._initialize(data)
        self._solve(data, self.theta, n_iterc=30)
        _, grad_funct = self._construct_f(data)
        x = grad_funct(self.theta)
        if self.forward.space:
            x = x.reshape(-1, self.forward.dc, x.shape[1])
            return np.linalg.norm(x, axis=1)
        return np.abs(x)

    @staticmethod
    def _residual(theta0: FloatArray, theta1: FloatArray) -> float:
        diff = theta1 - theta0
        num = diff ** 2
        den = theta0 ** 2
        if den.sum() <= 0:
            return np.inf
        else:
            return sqrt(num.sum() / den.sum())


@dataclass(frozen=True)
class ChampLassoFit(SolverFit):
    """Fitted state produced by :class:`ChampLasso`."""

    history: ChampLassoHistory
    gamma: list
    sigma_b: list[FloatArray]

    def evaluate_objective(
            self,
            forward: ForwardModel,
            data: RegressionData,
            return_weighted_l2: bool = False,
    ) -> float | tuple[float, float]:
        """Evaluate the ChampLasso likelihood objective on whitened data."""
        return _evaluate_objective(
            forward, self.theta, self.sigma_b, data, return_weighted_l2,
        )

    def score(
            self,
            forward: ForwardModel,
            data: RegressionData,
    ) -> dict[str, float]:
        """Likelihood objective on ``data`` and its weighted-L2 term."""
        cross_fit, weighted_l2_error = self.evaluate_objective(forward, data, True)
        return {'cross_fit': cross_fit, 'weighted_l2_error': weighted_l2_error}


def select_by_criterion(cv_results: Sequence[CVResult], criterion: str = 'cross-fit') -> ChampLasso:
    """Pick the best solver from cross-validation results by the given criterion.

    Parameters
    ----------
    cv_results
        Results to choose from.
    criterion
        Criterion for best fit. Possible values:

        - ``'cross-fit'``: The smallest cross-fit value (default)
        - ``'l2'``: The smallest l2 error
        - ``'l2/mu'``: The local minimum in the l2 error with smallest trf (largest mu)
    """
    if criterion == 'cross-fit':
        return min(cv_results, key=lambda result: result.scores['cross_fit']).solver
    elif criterion == 'l2':
        return min(cv_results, key=lambda result: result.scores['l2_error']).solver
    elif criterion == 'l2/mu':
        results = sorted(cv_results, key=lambda result: result.solver.mu)
        peaks, _ = find_peaks([-result.scores['l2_error'] for result in results])  # find local minima
        if len(peaks) > 0:
            # higher mu -> smaller trf
            return max((results[peak] for peak in peaks), key=lambda result: result.solver.mu).solver
        return min(results, key=lambda result: result.scores['l2_error']).solver
    else:
        raise ValueError(f'{criterion=}')


def _select_es_solver(cv_results: Sequence[CVResult], minimum_mu: float) -> ChampLasso | None:
    """Return the solver at the first ES local minimum above ``minimum_mu``."""
    results = sorted(cv_results, key=lambda result: result.solver.mu)
    for i, result in enumerate(results[:-1]):
        if result.solver.mu < minimum_mu:
            continue
        if result.scores['estimation_stability'] < results[i + 1].scores['estimation_stability']:
            return result.solver
    return None


@dataclass(frozen=True)
class ChampLasso(Solver):
    """Alternating FASTA/Champagne solver for NCRF estimation.

    Parameters
    ----------
    mu
        Regularizer parameter. A number fits one model, a sequence selects among
        an explicit grid with cross-validation, and ``'auto'`` derives and
        cross-validates a grid from the data (default).
    n_iter
        Number of outer iterations of the algorithm.
    n_iterc
        Number of Champagne iterations within each outer iteration.
    n_iterf
        Number of FASTA iterations within each outer iteration.
    tol
        Tolerance factor deciding stopping criterion for the overall algorithm.
        Iteration stops when ``norm(trf_new - trf_old)/norm(trf_old) < tol``.
    store_theta
        Store the ``theta`` estimate after each outer iteration in the solver
        history (default ``False``).
    store_gamma
        Store the source covariances after each outer iteration (default ``False``).
    store_sigma_b
        Store the data covariances after each outer iteration (default ``False``).

    """

    criterion = 'cross_fit'

    mu: MuArg = 'auto'
    n_iter: int = 30
    n_iterc: int = 10
    n_iterf: int = 100
    tol: float = 1e-5
    store_objective: bool = True
    store_residual: bool = True
    store_theta: bool = False
    store_gamma: bool = False
    store_sigma_b: bool = False

    def without_history(self) -> ChampLasso:
        """Disable all per-iteration storage for cross-validation folds."""
        return replace(self, **{field: False for field in _STORE_FIELDS})

    def select(
            self,
            cv_results: Sequence[CVResult],
            cv: CrossValidation,
    ) -> ChampLasso:
        """Select ``mu`` by cross-fit, optionally refined by estimation stability."""
        logger = logging.getLogger(__name__)
        solver = select_by_criterion(cv_results, 'cross-fit')
        if not cv.use_es:
            return solver

        if solver.mu == max(result.solver.mu for result in cv_results):
            logger.info(f'\nCVmu is {solver.mu}: could not find mu based on estimation stability criterion\nContinuing with cross-validation only.')
            return solver
        es_solver = _select_es_solver(cv_results, solver.mu)
        if es_solver is None:
            logger.warning('\nNo ES minima found: could not find mu based on estimation stability criterion.\nContinuing with cross-validation only.')
            return solver
        return es_solver

    def refine(
            self,
            candidates: Sequence[ChampLasso],
            best: ChampLasso,
    ) -> tuple[ChampLasso, ...]:
        """Return one additional decade when the best candidate is on a grid boundary."""
        logger = logging.getLogger(__name__)
        mus = [candidate.mu for candidate in candidates]
        if best.mu == min(mus):
            new_mus = np.logspace(np.log10(best.mu) - 1, np.log10(best.mu), 4)[:-1]
        elif best.mu == max(mus):
            new_mus = np.logspace(np.log10(best.mu), np.log10(best.mu) + 1, 4)[1:]
        else:
            return ()
        direction = 'left' if new_mus[-1] < best.mu else 'right'
        logger.info(f'CVmu is {best.mu}: extending range of mu towards {direction}')
        return tuple(replace(best, mu=float(mu)) for mu in new_mus)

    def cv_table(
            self,
            cv_results: Sequence[CVResult],
            selected: ChampLasso,
    ) -> fmtxt.Table:
        """Summarize cross-validation scores by ``mu``."""
        results = sorted(cv_results, key=lambda result: result.solver.mu)
        best_mu = {criterion: select_by_criterion(cv_results, criterion).mu for criterion in ('cross-fit', 'l2/mu')}

        table = fmtxt.Table('lllll')
        table.cells('mu', 'cross-fit', 'l2-error', 'weighted l2-error', 'ES metric')
        table.midrule()
        fmt = '%.5f'
        for result in results:
            table.cell(fmtxt.stat(result.solver.mu, fmt=fmt))
            table.cell(fmtxt.stat(result.scores['cross_fit'], fmt, 1 if result.solver.mu == best_mu['cross-fit'] else 0, 1))
            table.cell(fmtxt.stat(result.scores['l2_error'], fmt, 1 if result.solver.mu == best_mu['l2/mu'] else 0, 1))
            table.cell(fmtxt.stat(result.scores['weighted_l2_error'], fmt=fmt))
            table.cell(fmtxt.stat(result.scores['estimation_stability'], fmt=fmt))
        if selected.mu == min(result.solver.mu for result in results):
            table.caption("Warnings: Best mu is smallest mu")
        return table

    def candidates(
            self,
            forward: ForwardModel,
            data: RegressionData,
    ) -> tuple[ChampLasso, ...]:
        """Resolve ``mu`` into fixed solver configurations."""
        if isinstance(self.mu, float):
            return self,
        elif isinstance(self.mu, str):
            if self.mu == 'auto':
                return self.auto_candidates(forward, data)
            raise ValueError(f"mu={self.mu!r}: expected a number, a sequence of numbers, or 'auto'")
        elif isinstance(self.mu, Real) and not isinstance(self.mu, bool):
            return replace(self, mu=float(self.mu)),

        try:
            values = tuple(self.mu)
        except TypeError:
            raise TypeError(
                f"mu={self.mu!r}: expected a number, a sequence of numbers, "
                "or 'auto'",
            ) from None
        if not values:
            raise ValueError("mu grid must contain at least one value")
        if any(isinstance(value, (bool, str, bytes)) for value in values):
            raise TypeError(f"mu={self.mu!r}: all grid values must be numbers")
        try:
            values = tuple(float(value) for value in values)
        except (TypeError, ValueError):
            raise TypeError(
                f"mu={self.mu!r}: all grid values must be numbers",
            ) from None
        return tuple(replace(self, mu=value) for value in values)

    def solve(
            self,
            forward: ForwardModel,
            data: RegressionData,
            *,
            verbose: bool = False,
    ) -> ChampLassoFit:
        """Estimate NCRF weights for one prepared, whitened dataset."""
        if not isinstance(self.mu, Real) or isinstance(self.mu, bool):
            raise ValueError("ChampLasso.solve() requires a fixed numeric mu; use NCRF.fit() to resolve a grid or mu='auto'")
        mu = float(self.mu)
        history = ChampLassoHistory(**{field: getattr(self, field) for field in _STORE_FIELDS})
        state = _ChampLassoState(forward, self.n_iter, self.n_iterc, self.n_iterf)
        state.run(data, mu, self.tol, history, verbose)
        return ChampLassoFit(
            theta=state.theta,
            history=history,
            gamma=state.Gamma,
            sigma_b=state.Sigma_b,
        )

    def auto_candidates(
            self,
            forward: ForwardModel,
            data: RegressionData,
    ) -> tuple[ChampLasso, ...]:
        """Derive the standard seven-value ``mu`` grid from the data."""
        state = _ChampLassoState(forward, self.n_iter, self.n_iterc, self.n_iterf)
        gradient = state.gradient(data)
        hi = log10(np.percentile(gradient, 99.0))
        mus = np.logspace(hi - 2, hi, 7)
        return tuple(replace(self, mu=float(mu)) for mu in mus)
