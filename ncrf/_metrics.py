"""Solver-independent prediction metrics for fitted NCRF models."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._data import RegressionData
    from ._model import NCRFModel


def explained_variance(
        model: NCRFModel,
        data: RegressionData,
        *,
        accept_whitening: bool = False,
) -> float:
    """Mean sensor-by-segment explained variance in whitened sensor space."""
    data = model._whiten(data, accept_whitening)
    residual_ratio = 0.0
    for meg, covariate in data:
        residual = meg - model._predict_whitened(covariate)
        residual_ratio += (
            np.nansum(np.var(residual, axis=1) / np.var(meg, axis=1))
            / residual.shape[0]
        )
    return 1 - residual_ratio / len(data)


def l2_error(
        model: NCRFModel,
        data: RegressionData,
        *,
        accept_whitening: bool = False,
) -> float:
    """Mean unweighted squared prediction error in whitened sensor space."""
    data = model._whiten(data, accept_whitening)
    error = 0.0
    for meg, covariate in data:
        residual = meg - model._predict_whitened(covariate)
        error += 0.5 * (residual ** 2).sum()
    return error / len(data)
