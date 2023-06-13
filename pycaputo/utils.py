# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import pathlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Protocol, TypeVar, Union

import numpy as np
from typing_extensions import TypeAlias

from pycaputo.logging import get_logger

logger = get_logger(__name__)

# {{{ typing

#: A generic invariant :class:`typing.TypeVar`.
T = TypeVar("T")

#: A union of types supported as paths.
PathLike = Union[pathlib.Path, str]

if TYPE_CHECKING:
    Array: TypeAlias = np.ndarray[Any, Any]
    Scalar: TypeAlias = Union[np.generic, Array]
else:
    #: Array type alias for :class:`numpy.ndarray`.
    Array: TypeAlias = np.ndarray
    #: Scalar type alias (generally a value convertible to a :class:`float`).
    Scalar: TypeAlias = Union[np.generic, Array]


class ScalarFunction(Protocol):
    """A generic callable that can be evaluated at :math:`x`.

    .. automethod:: __call__
    """

    def __call__(self, x: Array) -> Array:
        """
        :arg x: a scalar or array at which to evaluate the function.
        """


class StateFunction(Protocol):
    r"""A generic callable for right-hand side functions
    :math:`\mathbf{f}(t, \mathbf{y})`.

    .. automethod:: __call__
    """

    def __call__(self, t: float, y: Array) -> Array:
        """
        :arg t: time at which to evaluate the function.
        :arg y: state vector value at which to evaluate the function.
        """


class ScalarStateFunction(Protocol):
    """A generic callable similar to :class:`StateFunction` that returns a
    scalar.

    .. automethod:: __call__
    """

    def __call__(self, t: float, y: Array) -> float:
        """
        :arg t: time at which to evaluate the function.
        :arg y: state vector value at which to evaluate the function.
        """


class CallbackFunction(Protocol):
    """A generic callback for evolution equations that can be used to modify
    the state.

    .. automethod:: __call__
    """

    def __call__(self, t: float, y: Array) -> bool:
        """
        :arg t: time at which to evaluate the function.
        :arg y: state vector value at which to evaluate the function.

        :returns: if *True*, hint to the algorithm that the evolution should be
            stopped.
        """


ArrayOrScalarFunction = Union[Array, ScalarFunction]

# }}}


# {{{ Estimated Order of Convergence (EOC)


@dataclass(frozen=True)
class EOCRecorder:
    """Keep track of all your *estimated order of convergence* needs."""

    #: A string identifier for the value which is estimated.
    name: str = "Error"
    #: An expected order of convergence, if any.
    order: float | None = None

    #: A list of ``(h, error)`` entries added from :meth:`add_data_point`.
    history: list[tuple[float, float]] = field(default_factory=list, repr=False)

    def add_data_point(self, h: Any, error: Any) -> None:
        """Add a data point to the estimation.

        Note that both *h* and *error* need to be convertible to a float.

        :arg h: abscissa, a value representative of the "grid size".
        :arg error: error at given *h*.
        """
        self.history.append((float(h), float(error)))

    @property
    def estimated_order(self) -> float:
        """Estimated order of convergence for currently available data. The
        order is estimated by least squares through the given data
        (see :func:`estimate_order_of_convergence`).
        """
        if not self.history:
            return np.nan

        h, error = np.array(self.history).T
        _, eoc = estimate_order_of_convergence(h, error)
        return eoc

    @property
    def max_error(self) -> float:
        """Largest error (in absolute value) in current data."""
        r = np.amax(np.array([error for _, error in self.history]))
        return float(r)

    def __str__(self) -> str:
        return stringify_eoc(self)


def estimate_order_of_convergence(x: Array, y: Array) -> tuple[float, float]:
    """Computes an estimate of the order of convergence in the least-square sense.
    This assumes that the :math:`(x, y)` pair follows a law of the form

    .. math::

        y = m x^p

    and estimates the constant :math:`m` and power :math:`p`.
    """
    assert x.size == y.size
    if x.size <= 1:
        raise RuntimeError("Need at least two values to estimate order.")

    eps = np.finfo(x.dtype).eps
    c = np.polyfit(np.log10(x + eps), np.log10(y + eps), 1)
    return 10 ** c[-1], c[-2]


def estimate_gliding_order_of_convergence(
    x: Array, y: Array, *, gliding_mean: int | None = None
) -> Array:
    assert x.size == y.size
    if x.size <= 1:
        raise RuntimeError("Need at least two values to estimate order.")

    if gliding_mean is None:
        gliding_mean = x.size

    npoints = x.size - gliding_mean + 1
    return np.array(
        [
            estimate_order_of_convergence(
                x[i : i + gliding_mean], y[i : i + gliding_mean] + 1.0e-16
            )
            for i in range(npoints)
        ],
        dtype=x.dtype,
    )


def flatten(iterable: Iterable[Iterable[T]]) -> tuple[T, ...]:
    from itertools import chain

    return tuple(chain.from_iterable(iterable))


def stringify_eoc(*eocs: EOCRecorder) -> str:
    r"""
    :arg eocs: an iterable of :class:`EOCRecorder`\ s that are assumed to have
        the same number of entries in their histories.
    :returns: a string representing the results in *eocs* in the
        GitHub Markdown format.
    """
    histories = [np.array(eoc.history).T for eoc in eocs]
    orders = [
        estimate_gliding_order_of_convergence(h, error, gliding_mean=2)
        for h, error in histories
    ]

    h = histories[0][0]
    ncolumns = 1 + 2 * len(eocs)
    nrows = h.size

    lines = []
    lines.append(("h", *flatten([(eoc.name, "EOC") for eoc in eocs])))

    lines.append((":-:",) * ncolumns)

    for i in range(nrows):
        values = flatten(
            [
                (
                    f"{error[i]:.6e}",
                    "---" if i == 0 else f"{order[i - 1, 1]:.3f}",
                )
                for (_, error), order in zip(histories, orders)
            ]
        )
        lines.append((f"{h[i]:.3e}", *values))

    lines.append(
        ("Overall", *flatten([("", f"{eoc.estimated_order:.3f}") for eoc in eocs]))
    )

    expected = flatten(
        [("", f"{eoc.order:.3f}") for eoc in eocs if eoc.order is not None]
    )
    if expected:
        lines.append(("Expected", *expected))

    widths = [max(len(line[i]) for line in lines) for i in range(ncolumns)]
    formats = ["{:%s}" % w for w in widths]

    return "\n".join(
        [
            " | ".join(fmt.format(value) for fmt, value in zip(formats, line))
            for line in lines
        ]
    )


# }}}


# {{{ matplotlib helpers


def check_usetex(*, s: bool) -> bool:
    try:
        import matplotlib
    except ImportError:
        return False

    if matplotlib.__version__ < "3.6.0":
        return bool(matplotlib.checkdep_usetex(s))

    # NOTE: simplified version from matplotlib
    # https://github.com/matplotlib/matplotlib/blob/ec85e725b4b117d2729c9c4f720f31cf8739211f/lib/matplotlib/__init__.py#L439=L456

    import shutil

    if not shutil.which("tex"):
        return False

    if not shutil.which("dvipng"):
        return False

    if not shutil.which("gs"):
        return False

    return True


def set_recommended_matplotlib(use_tex: bool | None = None) -> None:
    try:
        import matplotlib.pyplot as mp
    except ImportError:
        return

    if use_tex is None:
        use_tex = "GITHUB_REPOSITORY" not in os.environ and check_usetex(s=True)

    defaults = {
        "figure": {"figsize": (8, 8), "dpi": 300, "constrained_layout": {"use": True}},
        "text": {"usetex": use_tex},
        "legend": {"fontsize": 32},
        "lines": {"linewidth": 2, "markersize": 10},
        "axes": {
            "labelsize": 32,
            "titlesize": 32,
            "grid": True,
            # NOTE: preserve existing colors (the ones in "science" are ugly)
            "prop_cycle": mp.rcParams["axes.prop_cycle"],
        },
        "xtick": {"labelsize": 24, "direction": "inout"},
        "ytick": {"labelsize": 24, "direction": "inout"},
        "axes.grid": {"axis": "both", "which": "both"},
        "xtick.major": {"size": 6.5, "width": 1.5},
        "ytick.major": {"size": 6.5, "width": 1.5},
        "xtick.minor": {"size": 4.0},
        "ytick.minor": {"size": 4.0},
    }

    from contextlib import suppress

    with suppress(ImportError):
        import SciencePlots  # noqa: F401

    if "science" in mp.style.library:
        mp.style.use(["science", "ieee"])
    elif "seaborn-v0_8" in mp.style.library:
        # NOTE: matplotlib v3.6 deprecated all the seaborn styles
        mp.style.use("seaborn-v0_8-white")
    elif "seaborn" in mp.style.library:
        # NOTE: for older versions of matplotlib
        mp.style.use("seaborn-white")

    for group, params in defaults.items():
        with suppress(KeyError):
            mp.rc(group, **params)


@contextmanager
def figure(filename: PathLike | None = None, **kwargs: Any) -> Iterator[Any]:
    import matplotlib.pyplot as mp

    fig = mp.figure()
    try:
        yield fig
    finally:
        if filename is not None:
            savefig(fig, filename, **kwargs)
        else:
            mp.show()

        mp.close(fig)


def savefig(fig: Any, filename: PathLike, **kwargs: Any) -> None:
    import matplotlib.pyplot as mp

    filename = pathlib.Path(filename)
    if not filename.suffix:
        ext = mp.rcParams["savefig.format"]
        filename = filename.with_suffix(f".{ext}").resolve()

    logger.info("Saving '%s'", filename)

    fig.tight_layout()
    fig.savefig(filename, **kwargs)


# }}}
