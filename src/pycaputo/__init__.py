# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pycaputo.derivatives import (
    CaputoDerivative,
    FractionalOperator,
    RiemannLiouvilleDerivative,
    Side,
)
from pycaputo.differentiation import DerivativeMethod
from pycaputo.grid import Points
from pycaputo.logging import get_logger
from pycaputo.stepping import FractionalDifferentialEquationMethod
from pycaputo.typing import Array, ArrayOrScalarFunction, PathLike, ScalarFunction

if TYPE_CHECKING:
    from pycaputo.derivatives import FractionalOperatorT
    from pycaputo.typing import StateFunctionT

log = get_logger("pycaputo")


# {{{ diff


def diff(
    f: ArrayOrScalarFunction,
    p: Points,
    d: FractionalOperator | float,
) -> Array:
    """Compute the fractional-order derivative of *f* at the points *p*.

    :arg f: an array or callable to compute the derivative of. If this is an
        array, it is assumed that it is evaluated at every point in *p*.
    :arg p: a set of points at which to compute the derivative.
    :arg d: a fractional operator. If this is just a number, the standard
        Caputo derivative will be used.
    """
    import pycaputo.differentiation as fracd

    if d is None:
        raise ValueError("'d' is required if 'method is not given")

    if not isinstance(d, FractionalOperator):
        d = CaputoDerivative(d, side=Side.Left)

    m = fracd.guess_method_for_order(p, d)

    return fracd.diff(m, f, p)


# }}}


# {{{ quad


def quad(
    f: ArrayOrScalarFunction,
    p: Points,
    d: FractionalOperator | float,
) -> Array:
    """Compute the fractional-order integral of *f* at the points *p*.

    :arg f: an array or callable to compute the integral of. If this is an
        array, it is assumed that it is evaluated at every point in *p*.
    :arg p: a set of points at which to compute the integral.
    :arg d: a fractional operator. If this is just a number, the standard
        Riemann-Liouville integral will be used.
    """
    import pycaputo.quadrature as fracq

    if d is None:
        raise ValueError("'d' is required if 'method' is not given")

    if not isinstance(d, FractionalOperator):
        d = RiemannLiouvilleDerivative(d, side=Side.Left)

    m = fracq.guess_method_for_order(p, d)

    return fracq.quad(m, f, p)


# }}}


# {{{ grad


def grad(
    m: DerivativeMethod,
    f: ScalarFunction,
    p: Points,
    x: Array,
    a: Array | None = None,
) -> Array:
    """Compute the fractional-order gradient of *f* at the points *p*.

    The gradient is computed component by component using :func:`diff`. The
    arguments also have the same meaning.

    :arg p: a set of :class:`~pycaputo.grid.Points` on :math:`[0, 1]` that will
        be linearly transformed to use as a grid for computing the gradient.
        Essentially, this will do :math:`p_i = a_i + (x_i - a_i) * p`.
    :arg x: a set of points at which to compute the gradient.
    :arg a: a set of starting points of the fractional operator, which will be
        computed on :math:`[a_i, x_i]`.
    """
    # {{{ normalize inputs

    if a is None:
        a = np.zeros_like(x)

    if x.shape != a.shape:
        raise ValueError(
            f"Inconsistent values for 'x' and 'a': got shape {x.shape} points but"
            f" shape {a.shape} starts"
        )

    if any(x[i] <= a[i] for i in np.ndindex(x.shape)):
        raise ValueError("Lower limits 'a' must be smaller than 'x'")

    # }}}

    def make_component_f(i: tuple[int, ...]) -> ScalarFunction:
        x_r = x[..., None]
        e_i = np.zeros_like(x_r)
        e_i[i] = 1.0

        def f_i(y: Array) -> Array:
            return f(x_r + (y - x[i]) * e_i)

        return f_i

    def make_component_p(i: tuple[int, ...]) -> Points:
        return p.translate(a[i], x[i])

    import pycaputo.differentiation as fracd

    result = np.empty_like(x)
    for i in np.ndindex(x.shape):
        # FIXME: this should just compute the gradient at -1
        result[i] = fracd.diff(m, make_component_f(i), make_component_p(i))[-1]

    return result


# }}}


# {{{ fracevolve


@dataclass(frozen=True)
class Solution:
    """A solution to a fractional-order differential equation."""

    t: Array
    """Time steps at which the solution was approximated."""
    y: Array
    """Solution values at each time step."""


def fracevolve(
    m: FractionalDifferentialEquationMethod[FractionalOperatorT, StateFunctionT],
    *,
    dtinit: float | None = None,
    quiet: bool = False,
    log_per_step: int = 50,
) -> Solution:
    """Evolve the given method *m* to its final time and capture the solution.

    :arg dtinit: initial guess for the time step (see also
        :func:`~pycaputo.stepping.evolve`).
    :arg quiet: if *True*, suppress any output.
    :arg log_per_step: a number of steps at which to print logging output for the
        evolution. This can be completely disabled using *quiet*.
    """
    from pycaputo.events import StepAccepted
    from pycaputo.stepping import evolve
    from pycaputo.utils import TicTocTimer

    ts = []
    ys = []
    time = TicTocTimer()

    time.tic()
    for event in evolve(m, dtinit=dtinit):
        if isinstance(event, StepAccepted):
            if not quiet and event.iteration % log_per_step == 0:
                time.toc()
                log.info(
                    "%s norm %.12e (%s)",
                    event,
                    np.linalg.norm(event.y),
                    time.short(),
                )
                time.tic()

            ts.append(event.t)
            ys.append(event.y)

    return Solution(t=np.array(ts), y=np.array(ys).squeeze().T)


# }}}


# {{{ fracplot


_DEFAULT_XLABEL = ["t", "t", "x", "x"]
_DEFAULT_YLABEL = ["y", "y", "y", "y"]


def _get_default_dark(*, default: bool = False) -> tuple[tuple[bool, str], ...]:
    """Get combinations of light and dark flags.

    This function is meant to be used by the few example scripts used to generate
    figures for the documentation. They can be configured to generate both light
    and dark figures.

    :returns: a tuple of ``(flag, suffix)`` for each light / dark combo.
        This can be controlled by the ``PYCAPUTO_DARK`` environment variable,
        which can be set to *True*, *False* or ``"both"``. The suffix is just
        ``"-dark"`` or ``"-light"`` depending on the flag.
    """
    import os

    from pycaputo.utils import BOOLEAN_STATES

    if "PYCAPUTO_DARK" in os.environ:
        tmp = os.environ["PYCAPUTO_DARK"].lower().strip()
        if tmp == "both":
            return ((True, "-dark"), (False, "-light"))
        else:
            result = BOOLEAN_STATES.get(tmp, False)
            return ((result, "-dark" if result else "-light"),)
    else:
        return ((default, ""),)


def fracplot(
    sol: Solution,
    filename: PathLike | None = None,
    *,
    dark: bool | None = None,
    azimuth: float = -55.0,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot the solution of a fractional differential equation from :func:`fracevolve`.

    This function is very opinionated at the moment and mainly used for the
    examples. It can be extended, but the lower level functions, such as
    :func:`pycaputo.utils.figure` and direct calls to :mod:`matplotlib` should be
    used instead.

    :arg dark: if *True*, a dark themed plot is created instead.
    """
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        log.warning("'matplotlib' is not available.")
        return

    if filename is not None:
        filename = pathlib.Path(filename)

    t = sol.t
    y = sol.y
    dim = y.shape[0]
    if dim == 1 or y.ndim == 1:
        dim = 1

    if dim not in {1, 2, 3}:
        raise ValueError(f"Unsupported system dimension: {dim}")

    if xlabel is None:
        xlabel = _DEFAULT_XLABEL[dim]

    if ylabel is None:
        ylabel = _DEFAULT_YLABEL[dim]

    overrides = {"lines": {"linewidth": 1}} if dim == 3 else {}
    suffixes = _get_default_dark(default=bool(dark)) if dark is None else ((dark, ""),)
    outfile = None

    from pycaputo.utils import figure, set_recommended_matplotlib

    for dark_i, suffix_i in suffixes:
        if filename is not None:
            outfile = filename.parent / f"{filename.stem}{suffix_i}{filename.suffix}"

        set_recommended_matplotlib(dark=dark_i, overrides=overrides)

        if dim == 1:
            with figure(outfile) as fig:
                ax = fig.gca()

                ax.plot(t, y)
                ax.set_xlabel(f"${xlabel}$")
                ax.set_ylabel(f"${ylabel}$")
                if ylim is not None:
                    ax.set_ylim(ylim)
        elif dim == 2:
            with figure(outfile) as fig:
                ax = fig.gca()

                ax.plot(y[0], y[1])
                ax.set_xlabel(f"${xlabel}$")
                ax.set_ylabel(f"${ylabel}$")

                if ylim is not None:
                    ax.set_ylim(ylim)
        elif dim == 3:
            with figure(outfile, projection="3d") as fig:
                ax = fig.gca()
                ax.view_init(elev=15, azim=azimuth, roll=0)

                ax.plot(y[0], y[1], y[2])
                ax.set_xlabel(f"${xlabel}$")
                ax.set_ylabel(f"${ylabel}$")

                if dark_i:
                    # NOTE: force black facecolor in 3D to match the background
                    # of sphinx-book better in its dark mode
                    ax.set_facecolor("#111111")

                if ylim is not None:
                    ax.set_ylim(ylim)
        else:
            raise AssertionError


def fracplots(
    sol: Solution,
    filename: PathLike | None = None,
    *,
    dark: bool | None = None,
    xlabel: str | None = None,
    legend: str | Sequence[str] | None = None,
    ylim: tuple[float, float] | None = None,
    hlines: tuple[np.floating, ...] | None = None,
) -> None:
    """Plot the solution of a fractional differential equation from :func:`fracevolve`.

    Unlike :func:`fracplot`, this function plots all the components of the *sol*
    in a single plot vs time.

    :arg dark: if *True*, a dark themed plot is created instead.
    """
    if filename is not None:
        filename = pathlib.Path(filename)

    t = sol.t
    y = sol.y
    if y.ndim == 1:
        y = y.reshape(1, -1)
    dim = y.shape[0]

    if xlabel is None:
        xlabel = _DEFAULT_XLABEL[0]

    if legend is None:
        legend = [f"${_DEFAULT_YLABEL[0]}_{i}$" for i in range(dim)]

    if isinstance(legend, str):
        legend = [legend] * dim

    if len(legend) != dim:
        raise ValueError(
            "Legends do not match solution size: "
            f"got {len(legend)} legend labels for a {dim}d system"
        )

    suffixes = _get_default_dark(default=bool(dark)) if dark is None else ((dark, ""),)
    outfile = None

    from pycaputo.utils import figure, set_recommended_matplotlib

    for dark_i, suffix_i in suffixes:
        if filename is not None:
            outfile = filename.parent / f"{filename.stem}{suffix_i}{filename.suffix}"

        set_recommended_matplotlib(dark=dark_i)

        with figure(outfile) as fig:
            ax = fig.gca()

            if dim == 1:
                ax.plot(t, y[0])
                ax.set_ylabel(legend[0])
            else:
                for i in range(dim):
                    ax.plot(t, y[i], label=legend[i])

            if hlines is not None:
                for yh in hlines:
                    ax.axhline(yh, color="k", linestyle="--")

            ax.set_xlabel(f"${xlabel}$")
            if ylim is not None:
                ax.set_ylim(ylim)

            if dim > 1:
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=dim)


# }}}

__all__ = ("diff", "fracevolve", "fracplot", "fracplots", "grad", "quad")
