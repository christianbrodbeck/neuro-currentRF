from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._typing import FloatArray

if TYPE_CHECKING:
    from .._data import RegressionData
    from .._forward import ForwardModel


@dataclass(frozen=True)
class SolverFit:
    """Result of one solver execution; concrete fits can add diagnostics."""

    theta: FloatArray

    def evaluate_objective(
            self,
            forward: ForwardModel,
            data: RegressionData,
            return_weighted_l2: bool = False,
    ) -> float | tuple[float, float] | None:
        return None


class Solver(ABC):
    """Immutable configuration for an algorithm that estimates NCRF weights."""

    def candidates(
            self,
            forward: ForwardModel,
            data: RegressionData,
    ) -> tuple[Solver, ...]:
        """Return fixed configurations to compare before fitting."""
        return (self,)

    @abstractmethod
    def solve(
            self,
            forward: ForwardModel,
            data: RegressionData,
            *,
            verbose: bool = False,
    ) -> SolverFit:
        """Estimate source-space NCRF weights for prepared, whitened data."""
