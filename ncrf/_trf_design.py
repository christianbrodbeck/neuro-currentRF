"""The regression design: how stimuli map onto covariates, and back onto TRFs.

Estimation works in a compact Gabor-basis coefficient space (``theta``).
:class:`TRFDesign` holds the stimulus and basis metadata that defines this space:
:meth:`TRFDesign.from_stim` derives it from the predictors, and its layout
metadata is what :attr:`~ncrf._model.NCRF.h` uses to expand coefficients back into
response functions. It is small and picklable, and is stored both on
:class:`~ncrf._data.RegressionData` and on the fitted :class:`~ncrf._model.NCRF`,
so that response functions can be reconstructed without keeping the full
(typically much larger) dataset around.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from eelbrain import NDVar
import numpy as np

from ._linalg import gaussian_basis
from ._typing import FloatArray, IndexArray, ScaleArg, StimDimensions


def time_samples(times: Sequence[float], tstep: float) -> list[int]:
    """Convert TRF times in seconds to sample offsets."""
    return [int(round(t / tstep)) for t in times]


def filter_lengths(
        tstart: Sequence[float],
        tstop: Sequence[float],
        tstep: float,
) -> list[int]:
    """TRF length in samples, one value per predictor."""
    return [stop - start + 1 for start, stop in zip(time_samples(tstart, tstep), time_samples(tstop, tstep))]


def stim_dimensions(stim: Sequence[NDVar]) -> list[StimDimensions | None]:
    """Return the feature dimension of each predictor (``None`` when scalar).

    Parameters
    ----------
    stim
        Predictors for one segment. Each must have a ``time`` axis and may carry
        at most one feature dimension before time.
    """
    dims = []
    for x in stim:
        if x.ndim == 1:
            dims.append(None)
        elif x.ndim == 2:
            dim, _ = x.get_dims((None, 'time'))
            dims.append(dim)
        else:
            raise ValueError(f"stim {x!r}: more than 2 dimensions")
    return dims


# eq=False: the array fields make a generated __eq__ raise instead of returning a
# bool, and an identity comparison is what callers of assert_compatible() want.
@dataclass(frozen=True, eq=False, repr=False)
class TRFDesign:
    """Stimulus and basis metadata defining the coefficient space.

    Use :meth:`from_stim` to derive a design from predictor NDVars.

    Attributes
    ----------
    basis
        Gaussian basis matrices, one per predictor variable.
    tstart, tstep, tstop
        TRF timing; ``tstart``/``tstop`` hold one value per predictor.
    basis_std
        Standard deviation of the Gaussian basis functions in seconds.
    stim_is_single
        Whether the original stimulus input was a single predictor per segment;
        controls whether reconstruction returns a bare NDVar or a list.
    stim_dims
        Feature dimension for each predictor (``None`` for scalar predictors).
    stim_names
        Name of each predictor variable.
    stim_baseline, stim_scaling
        Centering and scaling applied to the covariates during data preparation,
        one value per expanded covariate channel, or ``None`` when that step was
        not applied. Set by :meth:`~ncrf._data.RegressionData.normalize`; a design
        that has them set describes covariates that carry them.
        :attr:`~ncrf._model.NCRF.h_scaled` uses :attr:`stim_scaling` to restore the
        original stimulus scale.
    scale
        Which scaling produced :attr:`stim_scaling` (``'l1'``, ``'l2'`` or
        ``'spectral'``), or ``None`` when the covariates were left unscaled.
    """

    basis: list[FloatArray]
    tstart: list[float]
    tstep: float
    tstop: list[float]
    basis_std: float
    stim_is_single: bool
    stim_dims: list[StimDimensions | None]
    stim_names: list[str]
    stim_baseline: FloatArray | None = None
    stim_scaling: FloatArray | None = None
    scale: ScaleArg = None

    @classmethod
    def from_stim(
            cls,
            stim: Sequence[NDVar],
            tstep: float,
            tstart: float | Sequence[float],
            tstop: float | Sequence[float],
            nlevel: int = 1,
            basis_std: float = 0.0085,
            stim_is_single: bool = False,
    ) -> TRFDesign:
        """Derive the design from the predictors of one segment.

        Parameters
        ----------
        stim
            Predictors for one segment, one NDVar per predictor variable. Each may
            be 1-D over time or carry one feature dimension before time.
        tstep
            Sample spacing in seconds.
        tstart
            Start of the TRF in seconds. A scalar applies to all predictors; a
            sequence specifies one start time per predictor.
        tstop
            Stop of the TRF in seconds. A scalar applies to all predictors; a
            sequence specifies one stop time per predictor.
        nlevel
            Density of Gabor basis atoms. Bigger → less dense. ``nlevel > 2``
            should be used with caution.
        basis_std
            Standard deviation of the Gaussian basis functions in seconds.
        stim_is_single
            Whether the original stimulus input was a single predictor per segment.
        """
        stim_dims = stim_dimensions(stim)
        tstart = list(tstart) if isinstance(tstart, Sequence) else [tstart] * len(stim_dims)
        tstop = list(tstop) if isinstance(tstop, Sequence) else [tstop] * len(stim_dims)
        if len(tstart) == 1:
            tstart = tstart * len(stim_dims)
        if len(tstop) == 1:
            tstop = tstop * len(stim_dims)
        if len(tstart) != len(stim_dims) or len(tstop) != len(stim_dims):
            raise ValueError(f"{tstart=}, {tstop=}: need one value per predictor ({len(stim_dims)})")

        basis = [gaussian_basis(int(round((fl - 1) / nlevel)), np.linspace(ts, te, fl), basis_std) for ts, te, fl in zip(tstart, tstop, filter_lengths(tstart, tstop, tstep))]
        return cls(
            basis=basis, tstart=tstart, tstep=tstep, tstop=tstop, basis_std=basis_std,
            stim_is_single=stim_is_single, stim_dims=stim_dims, stim_names=[x.name for x in stim],
        )

    def __repr__(self) -> str:
        predictors = tuple(self.stim_names)
        basis_counts = tuple(basis.shape[1] for basis in self.basis)
        lags = tuple(zip(self.tstart, self.tstop))
        tstep, basis_std, scale = self.tstep, self.basis_std, self.scale
        centered = self.stim_baseline is not None
        return f'<{type(self).__name__}: {predictors=}, {basis_counts=}, {lags=}, {tstep=}, {basis_std=}, {centered=}, {scale=}>'

    @property
    def stim_lens(self) -> list[int]:
        """Number of expanded covariate channels contributed by each predictor."""
        return [len(dim) if dim else 1 for dim in self.stim_dims]

    @property
    def start_samples(self) -> list[int]:
        """:attr:`tstart` in samples, one value per predictor."""
        return time_samples(self.tstart, self.tstep)

    @property
    def stop_samples(self) -> list[int]:
        """:attr:`tstop` in samples, one value per predictor."""
        return time_samples(self.tstop, self.tstep)

    @property
    def filter_length(self) -> list[int]:
        """TRF length in samples, one value per predictor."""
        return filter_lengths(self.tstart, self.tstop, self.tstep)

    @property
    def n_coefficients(self) -> int:
        """Total number of Gabor coefficients, i.e. the number of columns of ``theta``."""
        return sum(basis.shape[1] * n for basis, n in zip(self.basis, self.stim_lens))

    @property
    def basis_widths(self) -> IndexArray:
        """Number of basis functions of each expanded covariate channel."""
        return np.repeat([basis.shape[1] for basis in self.basis], self.stim_lens)

    @property
    def basis_column_sums(self) -> FloatArray:
        """Sum of each covariate column's basis function, one value per column.

        The covariate for a constant stimulus of 1, for rows whose full lag window
        lies inside the stimulus; i.e. the offset a unit :attr:`stim_baseline`
        introduces in each covariate column.
        """
        return np.concatenate([np.tile(basis.sum(0), n) for basis, n in zip(self.basis, self.stim_lens)])

    def expand(self, values: FloatArray) -> FloatArray:
        """Expand one value per covariate channel to one value per covariate column."""
        return np.repeat(values, self.basis_widths)

    def per_predictor(self, values: FloatArray) -> list[NDVar | float]:
        """Split one value per covariate channel into one item per predictor.

        Predictors with a feature dimension yield an :class:`~eelbrain.NDVar` over
        that dimension; scalar predictors yield a :class:`float`.
        """
        out = []
        i = 0
        for dim, n in zip(self.stim_dims, self.stim_lens):
            chunk = values[i:i + n]
            out.append(NDVar(chunk, (dim,)) if dim else float(chunk[0]))
            i += n
        return out

    def assert_compatible(self, other: TRFDesign) -> None:
        """Check that ``other`` describes the same coefficient space as this design.

        Only structural metadata is compared; the normalization a design records is
        the caller's business.

        Parameters
        ----------
        other
            Design to compare against.

        Raises
        ------
        ValueError
            If the designs describe different predictors, TRF timings, or bases.
        """
        if other is self:
            return
        for attr in ('stim_names', 'stim_dims', 'tstart', 'tstop', 'tstep', 'basis_std'):
            mine, theirs = getattr(self, attr), getattr(other, attr)
            if mine != theirs:
                raise ValueError(f"incompatible design: {attr} is {theirs} instead of {mine}")
        if len(other.basis) != len(self.basis) or not all(np.array_equal(a, b) for a, b in zip(self.basis, other.basis)):
            raise ValueError("incompatible design: different Gabor basis (check nlevel)")
