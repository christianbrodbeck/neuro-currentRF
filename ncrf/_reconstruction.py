"""Reconstruct response-function NDVars from fitted Gabor coefficients.

The estimation works in a compact Gabor-basis coefficient space (``theta``).
:class:`TRFDesign` holds the stimulus and basis metadata needed to expand those
coefficients back into labeled spatio-temporal response functions.  It is small
and picklable, and is stored on :class:`~ncrf._model.NCRFModel` so that response
functions can be reconstructed without keeping the full :class:`RegressionData`
around.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from eelbrain import NDVar, UTS
import numpy as np

from ._typing import FloatArray, StimDimensions

if TYPE_CHECKING:
    from ._forward import ForwardModel


@dataclass(repr=False)
class TRFDesign:
    """Stimulus and basis metadata needed to reconstruct response functions.

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
        Per-predictor centering/scaling applied during data preparation, used to
        restore the original stimulus scale in :meth:`reconstruct_scaled`.
    """

    basis: list[FloatArray]
    tstart: list[float]
    tstep: float
    tstop: list[float]
    basis_std: float
    stim_is_single: bool
    stim_dims: list[StimDimensions | None]
    stim_names: list[str]
    stim_baseline: Sequence[NDVar | float] | None
    stim_scaling: Sequence[NDVar | float] | None

    def __repr__(self) -> str:
        predictors = tuple(self.stim_names)
        basis_counts = tuple(basis.shape[1] for basis in self.basis)
        lags = tuple(zip(self.tstart, self.tstop))
        tstep, basis_std = self.tstep, self.basis_std
        return f'<{type(self).__name__}: {predictors=}, {basis_counts=}, {lags=}, {tstep=}, {basis_std=}>'

    def reconstruct(self, theta: FloatArray, forward: ForwardModel) -> NDVar | list[NDVar]:
        """Expand Gabor coefficients into spatio-temporal response-function NDVars."""
        source = forward.source
        space = forward.space
        n_vars = sum(len(dim) if dim else 1 for dim in self.stim_dims)
        if space:
            _shared_dims = (source, space)
        else:
            _shared_dims = (source, )

        if n_vars > 1:
            _trf = []
            start = 0
            stop = 0
            for basis, dim in zip(self.basis, self.stim_dims):
                stim_len = len(dim) if dim else 1
                stop += basis.shape[1] * stim_len
                block = theta[:, start:stop].copy()
                block = block.reshape((theta.shape[0], stim_len, -1))
                _trf.append(np.squeeze(block.swapaxes(1, 0)))
                start += basis.shape[1] * stim_len
        else:
            _trf = [theta]

        trf = [np.dot(x, basis.T) / forward.lead_field_scaling for x, basis in zip(_trf, self.basis)]

        h = []
        for x, dim, name, tstart in zip(trf, self.stim_dims, self.stim_names, self.tstart):
            if dim:
                time = UTS(tstart, self.tstep, x.shape[-1])
                shared_dims = (*_shared_dims, time)
                x = x.reshape((-1, *(map(len, shared_dims))))
                dims = (dim, *shared_dims)
            else:
                time = UTS(tstart, self.tstep, x.shape[-1])
                dims = (*_shared_dims, time)
                x = x.reshape(*(map(len, dims)))
            h.append(NDVar(x, dims, name=name))

        if self.stim_is_single:
            return h[0]
        else:
            return h

    def reconstruct_scaled(self, h: NDVar | list[NDVar]) -> NDVar | list[NDVar]:
        """Return ``h`` with the original stimulus scaling restored."""
        if self.stim_scaling is None:
            return h
        elif self.stim_is_single:
            return h * self.stim_scaling[0]
        else:
            return [x * s for x, s in zip(h, self.stim_scaling)]
