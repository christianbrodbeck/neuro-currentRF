"""Shared type aliases and numeric tolerances for the NCRF package."""
from __future__ import annotations

from typing import Callable, Literal, Sequence

from eelbrain import Categorial, Scalar, Space
import numpy as np
import numpy.typing as npt


_R_tol = np.finfo(np.float64).eps * 1e2
FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.int64]
TrialData = tuple[FloatArray, FloatArray]
ObjectiveFunction = Callable[[FloatArray], float]
GradientFunction = Callable[[FloatArray], FloatArray]
MuArg = float | Sequence[float] | FloatArray | Literal["auto"]
ScaleArg = Literal["l1", "l2", "spectral"] | None
StimDimensions = Categorial | Scalar | Space
