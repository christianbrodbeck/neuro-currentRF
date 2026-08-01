"""Prepared regression dataset and covariate construction for NCRF fitting.

``RegressionData`` turns Eelbrain objects into normalized numeric arrays with a
stable internal layout that the solver consumes directly. The layout itself is
described by the ``TRFDesign`` it carries.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from math import sqrt
from collections.abc import Iterator, Sequence

from eelbrain import NDVar, Sensor, UTS
import numpy as np
import numpy.typing as npt
from scipy import linalg

from ._trf_design import TRFDesign, stim_dimensions
from ._repr import _count_repr
from ._typing import FloatArray, IndexArray, ScaleArg, TrialData


SCALES = ('l1', 'l2', 'spectral')


def get_scaling(
        stim: Sequence[Sequence[NDVar]],
        design: TRFDesign,
        scale: ScaleArg,
) -> tuple[FloatArray, FloatArray | None]:
    """Stimulus centering and scaling values, one per expanded covariate channel.

    Parameters
    ----------
    stim
        Stimulus lists, one per segment; each inner list contains one NDVar per
        predictor.
    design
        Design describing the predictors.
    scale
        Compute the ``'l1'`` (mean absolute deviation) or ``'l2'`` (standard
        deviation) scale of each predictor. Any other value yields ``None`` for the
        scaling, since it is then not derived from the stimulus.

    Returns
    -------
    baseline
        The mean of each predictor.
    scaling
        The requested scale of each predictor, measured around its mean, or
        ``None``.

    Raises
    ------
    ValueError
        If a predictor channel is constant over time, and hence has no variation
        to scale by.
    """
    by_predictor = list(zip(*stim))  # -> [[stim_1_trial_1, stim_1_trial_2, ...], ...]
    # Check for constant (flat) predictors here, where the cause can be named
    constant = []
    for name, trials in zip(design.stim_names, by_predictor):
        arrays = [np.atleast_2d(t.x) for t in trials]  # (n_channels, n_times) per segment
        lo = np.min([x.min(1) for x in arrays], axis=0)
        hi = np.max([x.max(1) for x in arrays], axis=0)
        for i in np.flatnonzero(lo == hi):
            constant.append(name if len(lo) == 1 else f'{name}[{i}]')
    if constant:
        raise ValueError(f"{', '.join(constant)}: predictor is constant over time, so it has no variation to scale by; drop it, or prepare the data with scale=None")

    n = sum(len(x.time) for x in by_predictor[0])
    means = [sum(x.sum('time') for x in trials) / n for trials in by_predictor]
    baseline = _channel_values(means, design.stim_lens)

    # Scale by the variation around the mean, whether or not the covariates end up
    # centered; the raw magnitude would let a predictor's offset dominate its scale
    centered = [[x - mean for x in trials] for mean, trials in zip(means, by_predictor)]
    if scale == 'l1':
        scales = [sum(x.abs().sum('time') for x in trials) / n for trials in centered]
    elif scale == 'l2':
        scales = [(sum((x ** 2).sum('time') for x in trials) / n) ** 0.5 for trials in centered]
    else:  # 'spectral' is computed after covariate construction
        return baseline, None
    return baseline, _channel_values(scales, design.stim_lens)


def _channel_values(
        values: Sequence[NDVar | float],
        stim_lens: Sequence[int],
) -> FloatArray:
    """Flatten one value per predictor into one value per covariate channel."""
    return np.concatenate([x.x if isinstance(x, NDVar) else np.full(n, x) for x, n in zip(values, stim_lens)])


def _pending(
        current: FloatArray | None,
        target: FloatArray | None,
        name: str,
) -> FloatArray | None:
    """The ``target`` values still to be applied to covariates carrying ``current``."""
    if current is None:
        return target
    elif target is None or not np.array_equal(current, target):
        raise ValueError(f"data covariates already carry a different {name}; prepare the data with scale=None to apply a different normalization")
    return None


def _check_scaling(
        scaling: FloatArray,
        design: TRFDesign,
) -> None:
    """Check that scaling factors can be divided by without destroying the covariates.

    Dividing by a factor of 0 fills the covariate columns with NaN, and a negative
    or non-finite factor corrupts them just as silently, so a design carrying such
    factors must be rejected rather than applied.

    Parameters
    ----------
    scaling
        Scaling factor of each covariate channel.
    design
        Design the factors belong to, used to name the offending channels.

    Raises
    ------
    ValueError
        If any factor is not finite and strictly positive.
    """
    bad = ~(np.isfinite(scaling) & (scaling > 0))
    if not bad.any():
        return
    channels = [name if n == 1 else f'{name}[{i}]' for name, n in zip(design.stim_names, design.stim_lens) for i in range(n)]
    items = ', '.join(f'{channels[i]}={scaling[i]:g}' for i in np.flatnonzero(bad))
    raise ValueError(f"invalid {design.scale!r} scaling ({items}): scaling factors must be finite and > 0; a predictor that is constant over time has no variation to scale by")


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
    design
        The ``TRFDesign`` that ``covariates`` were built with. It also records the
        normalization that was applied to ``covariates`` (see :meth:`normalize`),
        which is what lets a model fitted on one dataset predict another.
    sensor_dim
        Sensor dimension shared by all MEG segments.
    is_whitened
        Whether ``meg`` has already been transformed by a whitening filter.
    """

    meg: list[FloatArray]  # (sensor, time)
    covariates: list[FloatArray]  # (time, covariate)
    norm_factor: float
    design: TRFDesign
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
            scale: ScaleArg = 'spectral',
            stim_is_single: bool = False,
            basis_std: float = 0.0085,
            in_place: bool = False,
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
        scale
            Normalization applied to the covariates. Each predictor's mean is
            subtracted, and each covariate channel is divided by one factor:

            - ``'spectral'`` (default): the channel's average spectral norm, which
              equalizes covariate scales across predictor variables.
            - ``'l1'``/``'l2'``: the predictor's mean absolute deviation or standard
              deviation.
            - ``None``: leave the covariates on their raw scale, without centering.

            Prepare data for prediction with ``scale=None`` and apply the fitted
            model's normalization with :meth:`normalize`.
        stim_is_single
            Whether the original stimulus input was a single predictor per segment.
        basis_std
            Standard deviation of the Gaussian basis functions in seconds.
        in_place
            If ``False`` (default), a copy of ``meg`` is made before it is rescaled.
            Set to ``True`` to modify it in place. ``stim`` is never modified.
        pad_stim
            If ``False`` (default), keep only rows whose full lag window is inside
            the stimulus time axis. If ``True``, retain edge rows with zero-padded
            covariates; with normalization those rows then correspond to a raw
            stimulus of 0 (rather than 0 after centering).
        """
        if not meg:
            raise ValueError("meg is empty")
        elif len(meg) != len(stim):
            raise ValueError("meg and stim have different lengths")
        elif scale is not None and scale not in SCALES:
            raise ValueError(f"{scale=}, need None or one of {SCALES}")

        # The design is fully determined by the first segment's predictors
        sensor_dim = meg[0].get_dim('sensor')
        first_time: UTS = meg[0].get_dim('time')
        tstep = first_time.tstep
        trial_length = len(first_time)
        design = TRFDesign.from_stim(stim[0], tstep, tstart, tstop, nlevel, basis_std, stim_is_single)

        row_slice = None
        if not pad_stim:
            # covariate_from_stim() fills the full MEG axis with zero-padded lag
            # histories. ``row_slice`` keeps only samples whose complete lag
            # window lies inside the stimulus.
            drop_start = max(0, *design.stop_samples)
            drop_stop = max(0, *(-s for s in design.start_samples))
            if drop_start or drop_stop:
                row_slice = slice(drop_start, -drop_stop if drop_stop else None)

        # Filter length/start per expanded covariate channel
        fl_rep = np.repeat(design.filter_length, design.stim_lens)
        st_rep = np.repeat(design.start_samples, design.stim_lens)

        meg_arrays: list[FloatArray] = []
        covariate_arrays: list[FloatArray] = []
        norm_factor = None

        for i_segment, (m, ss) in enumerate(zip(meg, stim)):
            if m.get_dim('sensor') != sensor_dim:
                raise ValueError(f'{meg=}: combining data segments with different sensor configurations is not supported')

            meg_time: UTS = m.get_dim('time')
            if meg_time.tstep != tstep:
                raise ValueError(f"{meg=}: segment {i_segment} time-step incompatible with first segment")
            if len(meg_time) != trial_length:
                raise NotImplementedError(f"{meg=}: unequal trial length")

            for x in ss:
                if x.get_dim('time') != meg_time:
                    raise ValueError(f"segment {i_segment} stim {x!r}: time axis incompatible with meg")
            if stim_dimensions(ss) != design.stim_dims:
                raise ValueError(f"{stim=}: segment {i_segment} dimensions incompatible with first segment")

            # Extract and normalize MEG array
            y = m.get_data(('sensor', 'time'))
            y_ = y.astype(np.float64, copy=False)
            y = y_ if (in_place or y_.base is None) else y_.copy()

            # Build basis-projected covariate matrix
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
            for l, b in zip(design.stim_lens, design.basis):
                covariates.extend([np.dot(x, b) / norm_factor for x in raw_covs[i:i + l]])
                i += l
            covariate_arrays.append(np.concatenate(covariates, axis=1).astype(np.float64))

        data = cls(meg_arrays, covariate_arrays, norm_factor, design, sensor_dim)

        if scale is not None:
            baseline, stim_scaling = get_scaling(stim, design, scale)
            # Center first, so that spectral norms are measured on centered covariates
            data = data.normalize(replace(design, stim_baseline=baseline))
            if scale == 'spectral':
                stim_scaling = data._spectral_norms()
            data = data.normalize(replace(data.design, stim_scaling=stim_scaling, scale=scale))
        return data

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
        predictors = tuple(self.design.stim_names)
        whitened = self.is_whitened
        return f"<{type(self).__name__}: {_count_repr(n_segments, 'segment')}, {_count_repr(n_sensors, 'sensor')}, {_count_repr(n_samples, 'sample')}/segment, {_count_repr(n_covariates, 'covariate')}, {predictors=}, {whitened=}>"

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

    def _spectral_norms(self) -> FloatArray:
        """Spectral norm of each covariate channel, averaged across segments."""
        splits = np.cumsum(self.design.basis_widths)[:-1]
        norms = [[linalg.norm(block, 2) for block in np.split(cov, splits, axis=1)] for cov in self.covariates]
        return np.array(norms).mean(axis=0)

    def normalize(self, design: TRFDesign) -> RegressionData:
        """Return a dataset carrying the centering and scaling recorded in ``design``.

        Normalization is a linear operation on the covariates, so applying it here
        is equivalent to applying it to the stimulus before covariate construction.
        Use this to prepare data for a fitted model, which can only be applied to
        covariates on the scale it was fit on::

            data = data.normalize(model.design)

        Parameters
        ----------
        design
            Design specifying the normalization to apply; it must describe the same
            coefficient space as this dataset's design. Steps this dataset already
            carries are skipped, so normalizing twice is a no-op.

        Notes
        -----
        The covariates are never modified in place. :meth:`whiten` hands out a
        dataset that shares covariate arrays with this one, and writing through
        them would leave the other dataset carrying a normalization that its own
        ``design`` does not record.

        Raises
        ------
        ValueError
            If ``design`` describes a different coefficient space, a different
            normalization than the covariates already carry, or a scaling factor
            that is not finite and strictly positive.
        """
        self.design.assert_compatible(design)
        baseline = _pending(self.design.stim_baseline, design.stim_baseline, 'baseline')
        scaling = _pending(self.design.stim_scaling, design.stim_scaling, 'scaling')
        if self.design.stim_scaling is not None and self.design.scale != design.scale:
            raise ValueError(f"data covariates carry {self.design.scale!r} scaling, the design specifies {design.scale!r}")
        if baseline is None and scaling is None:
            return replace(self, design=design)

        covariates = [cov.copy() for cov in self.covariates]
        if baseline is not None:
            # Every retained row has a full lag window, so subtracting a constant from
            # the stimulus offsets each covariate column by a constant.
            offset = design.expand(baseline) * design.basis_column_sums / self.norm_factor
            for cov in covariates:
                cov -= offset
        if scaling is not None:
            _check_scaling(scaling, design)
            factors = design.expand(scaling)
            for cov in covariates:
                cov /= factors
        return replace(self, covariates=covariates, design=design)

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
        return replace(self, meg=meg, is_whitened=True)

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
        return replace(self, meg=meg, covariates=covariates, norm_factor=norm_factor)
