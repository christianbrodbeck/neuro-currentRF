"""Prepared regression dataset and covariate construction for NCRF fitting.

``RegressionData`` turns Eelbrain objects into normalized numeric arrays with a
stable internal layout that the solver consumes directly.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
from functools import cached_property
from math import sqrt
from typing import Iterator, Sequence

from eelbrain import NDVar, Sensor, UTS
import numpy as np
import numpy.typing as npt
from scipy import linalg

from ._linalg import gaussian_basis
from ._reconstruction import TRFDesign
from ._repr import _count_repr
from ._typing import FloatArray, IndexArray, StimDimensions, TrialData


def covariate_from_stim(
        stims: Sequence[NDVar] | NDVar,
        Ms: Sequence[int] | npt.ArrayLike,
        starts: Sequence[int] | npt.ArrayLike,
) -> list[FloatArray]:
    """Form lagged covariate matrices from one or more stimulus NDVars.

    Parameters
    ----------
    stims
        Predictor variables. Each predictor must provide a ``time`` axis and may have
        at most one additional feature dimension before time.
    Ms
        Filter lengths, in samples, for each expanded predictor channel.
    starts
        Start offsets, in samples, for each expanded predictor channel.

    Returns
    -------
    list
        Covariate matrices, one per expanded predictor channel. Each matrix has
        one row per stimulus time sample; rows with incomplete stimulus history are
        zero-padded.
    """
    ws = []
    for stim in stims:
        if stim.ndim == 1:
            w = stim.get_data((np.newaxis, 'time'))
        else:
            dimnames = stim.get_dimnames(last='time')
            w = stim.get_data(dimnames)
        ws.append(w)
    ws = ws[0] if len(ws) == 1 else np.concatenate(ws, 0)
    assert len(ws) == len(Ms) == len(starts), f"Length of w ({len(ws)}), Ms ({len(Ms)}), and start ({len(starts)}) should be equal"

    n_times = ws.shape[1]
    Y = []
    for w, start, M in zip(ws, starts, Ms):
        X = np.zeros((n_times, M), dtype=w.dtype)
        for i in range(n_times):
            stop = i + 1
            start_i = max(0, stop - M)
            n = stop - start_i
            X[i, :n] = w[start_i:stop][::-1]
        if start != 0:
            # -ve tstart -> shift covariate matrix left
            # +ve tstart -> shift covariate matrix right
            X = np.roll(X, start, axis=0)
            if start < 0:
                X[start:] = 0
            else:
                X[:start] = 0
        Y.append(X)
    return Y


@dataclass(eq=False, repr=False)
class RegressionData:
    """Prepared dataset for NCRF fitting.

    Use :meth:`from_data` to construct a dataset from raw MEG and stimulus
    :class:`~eelbrain.NDVar` objects.

    Parameters
    ----------
    meg
        MEG signal arrays, one per segment, each shaped
        ``(n_sensors, n_times)``.
    covariates
        Basis-projected covariate matrices, one per segment, each shaped
        ``(n_times, n_basis_cols)``.
    norm_factor
        ``sqrt(n_times)`` of the first segment; used by :meth:`timeslice`
        to rescale sub-segments consistently.
    basis
        Gaussian basis matrices, one per predictor variable, each shaped
        ``(filter_length, n_basis)``.
    tstart
        TRF start time in seconds, one value per predictor.
    tstep
        Sample spacing in seconds, shared by all segments.
    tstop
        TRF stop time in seconds, one value per predictor.
    stim_is_single
        ``True`` when the original stimulus input contained a single
        predictor per segment rather than a list; controls whether
        :attr:`NCRF.h` returns a bare NDVar or a list.
    stim_dims
        Feature dimension for each predictor (``None`` for scalar predictors).
    stim_names
        Name of each predictor variable.
    baseline
        Per-predictor centering values subtracted before covariate
        construction, or ``None`` if no centering was applied.
    scaling
        Per-predictor scale factors applied after centering, or ``None``
        if no scaling was applied.
    stim_normalization
        Spectral norms of each predictor block before post-normalization,
        one inner list per segment.
    basis_std
        Standard deviation of the Gaussian basis functions in seconds.
    sensor_dim
        Sensor dimension shared by all MEG segments.
    is_whitened
        Whether ``meg`` has already been transformed by a whitening filter.
    """

    meg: list[FloatArray]  # (sensor, time)
    covariates: list[FloatArray]  # (time, covariate)
    norm_factor: float
    basis: list[FloatArray]  # (filter_time, covariate)
    tstart: list[float]
    tstep: float
    tstop: list[float]
    stim_is_single: bool
    stim_dims: list[StimDimensions | None]
    stim_names: list[str]
    baseline: Sequence[NDVar | float] | None
    scaling: Sequence[NDVar | float] | None
    stim_normalization: list[list[float]]  # (segment, expanded covariate)
    basis_std: float
    sensor_dim: Sensor
    is_whitened: bool = False

    def __post_init__(self) -> None:
        if len({m.shape[1] for m in self.meg}) > 1:
            raise NotImplementedError("Segments with unequal trial length")

    @classmethod
    def from_data(
            cls,
            meg: list[NDVar],
            stim: list[Sequence[NDVar]],
            tstart: float | Sequence[float],
            tstop: float | Sequence[float],
            nlevel: int = 1,
            baseline: Sequence[NDVar | float] | None = None,
            scaling: Sequence[NDVar | float] | None = None,
            stim_is_single: bool = False,
            basis_std: float = 0.0085,
            in_place: bool = False,
            post_normalize: bool = True,
            pad_stim: bool = False,
    ) -> RegressionData:
        """Construct a dataset from MEG and stimulus NDVars.

        Parameters
        ----------
        meg
            MEG segments, each an NDVar with ``sensor`` and ``time`` dimensions.
        stim
            Stimulus lists, one per segment; each inner list contains one NDVar per
            predictor. Each predictor may be 1-D over time or carry one feature
            dimension before time.
        tstart
            Start of the TRF in seconds. A scalar applies to all predictors; a
            sequence specifies one start time per predictor.
        tstop
            Stop of the TRF in seconds. A scalar applies to all predictors; a
            sequence specifies one stop time per predictor.
        nlevel
            Density of Gabor basis atoms. Bigger → less dense. ``nlevel > 2``
            should be used with caution.
        baseline
            Per-predictor means to subtract from ``stim`` before covariate
            construction.
        scaling
            Per-predictor scaling factors applied after baseline subtraction.
        stim_is_single
            Whether the original stimulus input was a single predictor per segment.
        basis_std
            Standard deviation of the Gaussian basis functions in seconds.
        in_place
            If ``False`` (default), copies of ``stim`` are made before applying
            baseline subtraction or scaling. Set to ``True`` to modify in place.
        post_normalize
            If ``True`` (default), equalize covariate scales across predictor
            blocks by dividing each block by its average spectral norm.
        pad_stim
            If ``False`` (default), keep only rows whose full lag window is inside
            the stimulus time axis. If ``True``, retain edge rows with zero-padded
            covariates.
        """
        if not meg:
            raise ValueError("meg is empty")
        elif len(meg) != len(stim):
            raise ValueError("meg and stim have different lengths")

        tstart = list(tstart) if isinstance(tstart, collections.abc.Sequence) else [tstart]
        tstop = list(tstop) if isinstance(tstop, collections.abc.Sequence) else [tstop]

        # State initialized from the first segment and compared to subsequent segments
        sensor_dim = None
        stim_dims = None
        stim_names = None
        tstep = None
        basis = None
        filter_length = None
        row_slice = None
        start_samples = None  # in-samples offsets, local to from_data

        meg_arrays: list[FloatArray] = []
        covariate_arrays: list[FloatArray] = []
        s_normalization = []
        trial_length = None
        norm_factor = None

        for i_segment, (m, ss) in enumerate(zip(meg, stim)):
            if in_place:
                ss = list(ss)
            else:
                ss = [s.copy() for s in ss]

            # Sensor dim
            if sensor_dim is None:
                sensor_dim = m.get_dim('sensor')
            elif m.get_dim('sensor') != sensor_dim:
                raise ValueError(f'{meg=}: combining data segments with different sensor configurations is not supported')

            # Time dim
            meg_time: UTS = m.get_dim('time')
            if tstep is None:
                tstep = meg_time.tstep
            elif meg_time.tstep != tstep:
                raise ValueError(f"{meg=}: segment {i_segment} time-step incompatible with first segment")
            if trial_length is None:
                trial_length = len(meg_time)
            elif len(meg_time) != trial_length:
                raise NotImplementedError(f"{meg=}: unequal trial length")

            # Determine stim feature dims for this segment
            cur_stim_dims = []
            for x in ss:
                if x.get_dim('time') != meg_time:
                    raise ValueError(f"segment {i_segment} stim {x!r}: time axis incompatible with meg")
                elif x.ndim == 1:
                    cur_stim_dims.append(None)
                elif x.ndim == 2:
                    dim, _ = x.get_dims((None, 'time'))
                    cur_stim_dims.append(dim)
                else:
                    raise ValueError(f"Segment {i_segment} stim {x!r}: more than 2 dimensions")

            if stim_dims is None:
                # Initialize time/basis parameters from the first segment
                stim_dims = cur_stim_dims
                stim_names = [x.name for x in ss]
                if len(tstart) == 1:
                    tstart = tstart * len(stim_dims)
                if len(tstop) == 1:
                    tstop = tstop * len(stim_dims)
                assert len(tstart) == len(stim_dims)
                assert len(tstop) == len(stim_dims)
                start_samples = [int(round(ts / tstep)) for ts in tstart]
                stop_samples = [int(round(te / tstep)) for te in tstop]
                filter_length = np.subtract(stop_samples, start_samples) + 1
                basis = []
                for ts, te, fl in zip(tstart, tstop, filter_length):
                    x = np.linspace(ts, te, fl)
                    basis.append(gaussian_basis(int(round((fl - 1) / nlevel)), x, basis_std))
                if not pad_stim:
                    # covariate_from_stim() fills the full MEG axis with
                    # zero-padded lag histories. ``row_slice`` keeps only samples
                    # whose complete lag window lies inside the stimulus.
                    drop_start = max(0, *stop_samples)
                    drop_stop = max(0, *(-s for s in start_samples))
                    if drop_start or drop_stop:
                        row_slice = slice(drop_start, -drop_stop if drop_stop else None)
            elif cur_stim_dims != stim_dims:
                raise ValueError(f"{stim=}: segment {i_segment} dimensions incompatible with first segment")

            # Apply stim normalization
            if baseline is not None:
                if len(baseline) != len(ss):
                    raise ValueError(f"baseline length {len(baseline)} != number of predictors {len(ss)}")
                for s, b in zip(ss, baseline):
                    s -= b
            if scaling is not None:
                if len(scaling) != len(ss):
                    raise ValueError(f"scaling length {len(scaling)} != number of predictors {len(ss)}")
                for s, sc in zip(ss, scaling):
                    s /= sc

            # Extract and normalize MEG array
            y = m.get_data(('sensor', 'time'))
            y_ = y.astype(np.float64, copy=False)
            y = y_ if (in_place or y_.base is None) else y_.copy()

            # Build basis-projected covariate matrix
            stim_lens = [len(d) if d else 1 for d in stim_dims]
            fl_rep = np.repeat(np.asanyarray(filter_length), stim_lens)
            st_rep = np.repeat(np.asanyarray(start_samples), stim_lens)
            raw_covs = covariate_from_stim(ss, fl_rep, st_rep)

            if row_slice is not None:
                y = y[:, row_slice]
                raw_covs = [x[row_slice] for x in raw_covs]
            if not y.shape[1]:
                raise ValueError(f"{meg=}: no samples remain after applying lag-validity crop")
            flat = np.var(y, axis=1) == 0
            if flat.any():
                raise ValueError(f"{meg=}: segment {i_segment} has flat channels ({', '.join(sensor_dim.names[flat])})")
            norm_factor = sqrt(y.shape[1])
            y /= norm_factor
            meg_arrays.append(y)

            i = 0
            covariates = []
            for d, b in zip(stim_dims, basis):
                l = len(d) if d else 1
                covariates.extend([np.dot(x, b) / sqrt(y.shape[1]) for x in raw_covs[i:i + l]])
                i += l
            s_normalization.append([linalg.norm(x, 2) for x in covariates])
            covariate_arrays.append(np.concatenate(covariates, axis=1).astype(np.float64))

        # Equalize covariate scales across predictor blocks
        if post_normalize:
            n_vars = sum(len(d) if d else 1 for d in stim_dims)
            if n_vars > 1:
                stim_lens = [len(d) if d else 1 for d in stim_dims]
                bl_lengths = np.repeat([b.shape[1] for b in basis], stim_lens)
                avg_norm = np.array(s_normalization).mean(axis=0)
                col = 0
                for bl, norm in zip(bl_lengths, avg_norm):
                    for cov in covariate_arrays:
                        cov[:, col:col + bl] /= norm
                    col += bl

        return cls(
            meg_arrays, covariate_arrays, norm_factor,
            basis=basis,
            tstart=tstart, tstep=tstep, tstop=tstop,
            stim_is_single=stim_is_single, stim_dims=stim_dims, stim_names=stim_names,
            baseline=baseline, scaling=scaling,
            stim_normalization=s_normalization, basis_std=basis_std,
            sensor_dim=sensor_dim,
        )

    def __iter__(self) -> Iterator[TrialData]:
        return zip(self.meg, self.covariates)

    def __len__(self) -> int:
        return len(self.meg)

    def __repr__(self) -> str:
        n_segments = len(self.meg)
        if self.meg:
            n_sensors, n_samples = self.meg[0].shape
        else:
            n_sensors, n_samples = len(self.sensor_dim), 0
        n_covariates = self.covariates[0].shape[1] if self.covariates else 0
        predictors = tuple(self.stim_names)
        whitened = self.is_whitened
        return f"<{type(self).__name__}: {_count_repr(n_segments, 'segment')}, {_count_repr(n_sensors, 'sensor')}, {_count_repr(n_samples, 'sample')}/segment, {_count_repr(n_covariates, 'covariate')}, {predictors=}, {whitened=}>"

    @property
    def trf_design(self) -> TRFDesign:
        """Small, picklable metadata bundle needed to reconstruct response functions.

        Stored on the fitted :class:`~ncrf._model.NCRFModel` (which persists to
        disk) so that ``h`` can be reconstructed without keeping the full,
        typically much larger, :class:`RegressionData`.
        """
        return TRFDesign(
            basis=self.basis,
            tstart=self.tstart, tstep=self.tstep, tstop=self.tstop, basis_std=self.basis_std,
            stim_is_single=self.stim_is_single, stim_dims=self.stim_dims, stim_names=self.stim_names,
            stim_baseline=self.baseline, stim_scaling=self.scaling,
        )

    @cached_property
    def bbt(self) -> list[FloatArray]:
        """Per-segment ``B @ B.T`` matrices for stored MEG arrays."""
        return [np.dot(b, b.T) for b in self.meg]

    @cached_property
    def bE(self) -> list[FloatArray]:
        """Per-segment ``B @ E`` cross-product matrices."""
        return [np.dot(b, E) for b, E in zip(self.meg, self.covariates)]

    @cached_property
    def EtE(self) -> list[FloatArray]:
        """Per-segment ``E.T @ E`` covariate Gram matrices."""
        return [np.dot(E.T, E) for E in self.covariates]

    def whiten(
            self,
            whitening_filter: FloatArray,
            accept_whitening: bool = False,
    ) -> RegressionData:
        """Return a dataset with MEG whitened.

        Parameters
        ----------
        whitening_filter
            Whitening matrix.
        accept_whitening
            Return an already-whitened dataset unchanged. The caller is
            responsible for ensuring that the right whitening filter was applied.

        Notes
        -----
        Uses shallow copies of unmodified data. If ``accept_whitening`` is true
        and the data is already whitened, returns this dataset unchanged.

        Raises
        ------
        ValueError
            If the dataset is already whitened and ``accept_whitening`` is false.
            Whitening twice is not equivalent to whitening once with the second
            filter (``W₂ @ W₁ @ meg ≠ W₂ @ meg``).
        """
        if self.is_whitened:
            if accept_whitening:
                return self
            raise ValueError("data is already whitened; pass accept_whitening=True to accept it")
        meg = [np.dot(whitening_filter, m) for m in self.meg]
        return RegressionData(
            meg, self.covariates, self.norm_factor,
            basis=self.basis,
            tstart=self.tstart, tstep=self.tstep, tstop=self.tstop,
            stim_is_single=self.stim_is_single, stim_dims=self.stim_dims,
            stim_names=self.stim_names, baseline=self.baseline, scaling=self.scaling,
            stim_normalization=self.stim_normalization,
            basis_std=self.basis_std, sensor_dim=self.sensor_dim,
            is_whitened=True,
        )

    def timeslice(self, idx: Sequence[int] | IndexArray) -> RegressionData:
        """Return a new dataset restricted to selected time indices.

        If this dataset ``.is_whitened``, the returned dataset is also
        marked as whitened and quadratic forms are recomputed lazily.

        Parameters
        ----------
        idx
            Integer indices selecting the time samples to retain.
        """
        norm_factor = sqrt(len(idx))
        mul = self.norm_factor / norm_factor
        meg = [m[:, idx] * mul for m in self.meg]
        covariates = [c[idx, :] * mul for c in self.covariates]
        return RegressionData(
            meg, covariates, norm_factor,
            basis=self.basis,
            tstart=self.tstart, tstep=self.tstep, tstop=self.tstop,
            stim_is_single=self.stim_is_single, stim_dims=self.stim_dims,
            stim_names=self.stim_names, baseline=self.baseline, scaling=self.scaling,
            stim_normalization=self.stim_normalization,
            basis_std=self.basis_std, sensor_dim=self.sensor_dim,
            is_whitened=self.is_whitened,
        )
