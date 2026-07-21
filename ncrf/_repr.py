"""Small helpers for concise object representations."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._forward import ForwardModel


def _count_repr(count: int, singular: str, plural: str | None = None) -> str:
    """Format a count with the appropriate singular or plural noun."""
    noun = singular if count == 1 else (plural or f'{singular}s')
    return f'{count} {noun}'


def _forward_summary(forward: ForwardModel) -> str:
    """User-relevant dimensions of a forward model."""
    orientation = 'free' if forward.space else 'fixed'
    return f"{_count_repr(len(forward.source), 'source')}, {_count_repr(len(forward.sensor), 'sensor')}, {orientation} orientation"
