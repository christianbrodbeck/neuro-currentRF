"""Sparsity penalties and their proximal operators used by the FASTA solver."""
from __future__ import annotations

import numpy as np

from . import opt
from ._typing import FloatArray


def g(x: FloatArray, mu: float) -> float:
    """Vector l1-norm penalty."""
    return mu * np.sum(np.abs(x))


def proxg(x: FloatArray, mu: float, tau: float) -> FloatArray:
    """Proximal operator for the l1-norm penalty."""
    return shrink(x, mu * tau)


def shrink(x: FloatArray, mu: float) -> FloatArray:
    """Soft-thresholding operator."""
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - mu, 0))


def g_group(x: FloatArray, mu: float) -> float:
    r"""group (l12) norm penalty:

            gg(x) = \sum ||x_s_{i,t}||

    where s_{i,t} = {x_{j,t}: j = 1*dc:(i+1)*dc}, i \in {1,2,...,#sources}, t \in {1,2,...,M}

    The three orientation components per source (fixed ``dc == 3``) are grouped
    along a reshaped view, so the caller's array is not modified.
    """
    l = x.shape[1]
    x3 = x.reshape(-1, 3, l)
    return mu * np.sqrt((x3 ** 2).sum(axis=1)).sum()


def proxg_group_opt(z: FloatArray, mu: float) -> FloatArray:
    """proximal operator for gg(x):

            prox_{mu gg}(x) = min  gg(z) + 1/ (2 * mu) ||x-z|| ** 2
                    x_s = max(1 - mu/||z_s||, 0) z_s

    Wrapper for the Cython kernel. The three orientation components per source
    (fixed ``dc == 3``) are grouped along a reshaped array. For a contiguous ``z``
    that reshape is a view, so the shrinkage overwrites ``z`` and the return value
    shares its buffer; otherwise it is a copy and ``z`` is left unchanged. Callers
    must therefore treat ``z`` as consumed and use only the return value.
    """
    l = z.shape[1]
    z3 = z.reshape(-1, 3, l)
    opt.cproxg_group(z3, mu, z3)
    return z3.reshape(-1, l)
