# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.utils import Array, StateFunction

logger = get_logger(__name__)


# {{{ root solve


def _solve_scalar(
    f: StateFunction,
    df: StateFunction | None,
    t: float,
    y0: Array,
    c: Array,
    r: Array,
    **kwargs: Any,
) -> Array:
    import scipy.optimize as so

    def func(y: Array) -> Array:
        return np.array(y - c * f(t, y) - r)

    def jac(y: Array) -> Array:
        assert df is not None
        return 1 - c * df(t, y)

    result = so.root_scalar(
        f=func,
        x0=y0,
        fprime=jac if df is not None else None,
        **kwargs,
    )

    return np.array(result.root)


def _solve_vector(
    f: StateFunction,
    df: StateFunction | None,
    t: float,
    y0: Array,
    c: Array,
    r: Array,
    **kwargs: Any,
) -> Array:
    import scipy.optimize as so

    def func(y: Array) -> Array:
        return np.array(y - c * f(t, y) - r, dtype=y0.dtype)

    def jac(y: Array) -> Array:
        assert df is not None
        return np.array(np.eye(y.size, dtype=y0.dtype) - np.diag(c) @ df(t, y))

    result = so.root(
        func,
        y0,
        jac=jac if df is not None else None,
        **kwargs,
    )

    return np.array(result.x)


def solve(
    f: StateFunction,
    df: StateFunction | None,
    t: float,
    y0: Array,
    c: Array,
    r: Array,
    **kwargs: Any,
) -> Array:
    r"""Solves an implicit update formula.

    This function is meant to solve implicit equations of the form

    .. math::

        \mathbf{y}_{n + 1} = \sum_{k = 0}^{n + 1} c_k \mathbf{f}(t_k, \mathbf{y}_k).

    Rearranging the implicit terms, we can write

    .. math::

        \mathbf{y}_{n + 1} - c_{n + 1} \mathbf{f}(t_{n + 1}, \mathbf{y}_{n + 1})
        = \mathbf{r}_n,

    and solve for the solution :math:`\mathbf{y}_{n + 1}`, where :math:`\mathbf{r}_n`
    contains all the explicit terms. This is done by a root finding algorithm
    provided by :func:`scipy.optimize.root`.

    :arg f: right-hand side function.
    :arg df: first-derivative (gradient or jacobian) of the right-hand side function
        with respect to :math:`\mathbf{y}`.
    :arg t: time at which the solution *y* is evaluated.
    :arg y0: initial guess at the unknown solution at time *t* (this is usually
        taken as :math:`\mathbf{y}_n`)
    :arg c: constant for the source term :math:`\mathbf{f}` that corresponds to
        the :attr:`~pycaputo.stepping.FractionalDifferentialEquationMethod.source`
        term of an evolution equation.
    :arg r: right-hand side term.

    :returns: solution :math:`\mathbf{y}^*_{n + 1}` of the above root finding
        problem.
    """

    if y0.size == 1:
        return _solve_scalar(f, df, t, y0, c, r, **kwargs)
    else:
        return _solve_vector(f, df, t, y0, c, r, **kwargs)


# }}}
