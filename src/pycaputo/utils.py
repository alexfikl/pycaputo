# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import os
import pathlib
import time
from contextlib import contextmanager, suppress
from dataclasses import Field, dataclass, field, is_dataclass
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np

try:
    # NOTE: available in Python >=3.10
    from typing import Concatenate, ParamSpec
except ImportError:
    from typing_extensions import Concatenate, ParamSpec  # type: ignore[assignment]

from pycaputo.logging import get_logger

logger = get_logger(__name__)


# {{{ typing

P = ParamSpec("P")

T = TypeVar("T")
"""A generic invariant :class:`typing.TypeVar`."""
R = TypeVar("R")
"""A generic invariant :class:`typing.TypeVar`."""
PathLike = Union[pathlib.Path, str]
"""A union of types supported as paths."""

if TYPE_CHECKING:
    Array = np.ndarray[Any, Any]
    Scalar = Union[np.generic, Array]
else:
    Array = np.ndarray
    """Array type alias for :class:`numpy.ndarray`."""
    Scalar = Union[np.generic, Array]
    """Scalar type alias (generally a value convertible to a :class:`float`)."""


class DataclassInstance(Protocol):
    """Dataclass protocol from
    `typeshed <https://github.com/python/typeshed/blob/770724013de34af6f75fa444cdbb76d187b41875/stdlib/_typeshed/__init__.pyi#L329-L334>`__."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


class ScalarFunction(Protocol):
    """A generic callable that can be evaluated at :math:`x`.

    .. automethod:: __call__
    """

    def __call__(self, x: Array, /) -> Array:
        """
        :arg x: a scalar or array at which to evaluate the function.
        """


class StateFunction(Protocol):
    r"""A generic callable for right-hand side functions
    :math:`\mathbf{f}(t, \mathbf{y})`.

    .. automethod:: __call__
    """

    def __call__(self, t: float, y: Array, /) -> Array:
        """
        :arg t: time at which to evaluate the function.
        :arg y: state vector value at which to evaluate the function.
        """


class ScalarStateFunction(Protocol):
    """A generic callable similar to :class:`StateFunction` that returns a
    scalar.

    .. automethod:: __call__
    """

    def __call__(self, t: float, y: Array, /) -> float:
        """
        :arg t: time at which to evaluate the function.
        :arg y: state vector value at which to evaluate the function.
        """


@runtime_checkable
class DifferentiableScalarFunction(Protocol):
    """A :class:`ScalarFunction` that can also compute its integer order derivatives.

    By default no derivatives are implemented, so subclasses can handle any such
    cases.
    """

    def __call__(self, x: Array, /, d: int = 0) -> Array:
        """Evaluate the function or any of its derivatives.

        :arg x: a scalar or array at which to evaluate the function.
        :arg d: order of the derivative.
        """


StateFunctionT = TypeVar("StateFunctionT", bound=StateFunction)
"""An invariant :class:`~typing.TypeVar` bound to :class:`StateFunction`."""

ArrayOrScalarFunction = Union[Array, ScalarFunction, DifferentiableScalarFunction]
"""A union of scalar functions."""

# fmt: off
BOOLEAN_STATES = {
    1: True, "1": True, "yes": True, "true": True, "on": True, "y": True,
    0: False, "0": False, "no": False, "false": False, "off": False, "n": False,
}
# fmt: on

# }}}


# {{{ Estimated Order of Convergence (EOC)


@dataclass(frozen=True)
class EOCRecorder:
    """Keep track of all your *estimated order of convergence* needs."""

    name: str = "Error"
    """A string identifier for the value which is estimated."""
    order: float | None = None
    """An expected order of convergence, if any."""

    history: list[tuple[float, float]] = field(default_factory=list, repr=False)
    """A list of ``(h, error)`` entries added from :meth:`add_data_point`."""

    @classmethod
    def from_data(
        cls, name: str, h: Array, error: Array, *, order: float | None = None
    ) -> EOCRecorder:
        eoc = cls(name=name, order=order)
        for i in range(h.size):
            eoc.add_data_point(h[i], error[i])

        return eoc

    def add_data_points(self, h: Array, error: Array) -> None:
        """Add multiple data points using :meth:`add_data_point`."""
        for h_i, e_i in zip(h, error):
            self.add_data_point(h_i, e_i)

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
        values = flatten([
            (
                f"{error[i]:.6e}",
                "---" if i == 0 else f"{order[i - 1, 1]:.3f}",
            )
            for (_, error), order in zip(histories, orders)
        ])
        lines.append((f"{h[i]:.3e}", *values))

    lines.append((
        "Overall",
        *flatten([("", f"{eoc.estimated_order:.3f}") for eoc in eocs]),
    ))

    if any(eoc.order is not None for eoc in eocs):
        expected = flatten([
            (("", f"{eoc.order:.3f}") if eoc.order is not None else ("", "--"))
            for eoc in eocs
        ])

        lines.append(("Expected", *expected))

    widths = [max(len(line[i]) for line in lines) for i in range(ncolumns)]
    formats = ["{:%s}" % w for w in widths]  # noqa: UP031

    return "\n".join([
        " | ".join(fmt.format(value) for fmt, value in zip(formats, line))
        for line in lines
    ])


def visualize_eoc(
    filename: pathlib.Path,
    *eocs: EOCRecorder,
    order: float | None = None,
    abscissa: str | Literal[False] = "h",
    ylabel: str | Literal[False] = "Error",
    olabel: str | Literal[False] | None = None,
    enable_legend: bool = True,
    overwrite: bool = True,
) -> None:
    """Plot the given :class:`EOCRecorder` instances in a loglog plot.

    :arg filename: output file name for the figure.
    :arg order: expected order for all the errors recorded in *eocs*.
    :arg abscissa: name for the abscissa.
    """
    if not eocs:
        raise ValueError("no EOCRecorders are provided")

    if order is not None and order <= 0.0:
        raise ValueError(f"The 'order' should be a non-negative real number: {order}")

    markers = ["o", "v", "^", "<", ">", "x", "+", "d", "D"]
    with figure(filename, overwrite=overwrite) as fig:
        ax = fig.gca()

        # {{{ plot eocs

        line = None
        for eoc, marker in zip(eocs, markers):
            h, error = np.array(eoc.history).T
            ax.loglog(h, error, marker=marker, label=eoc.name)

            imax = np.argmax(h)
            max_h = h[imax]
            max_e = error[imax]
            min_e = np.min(error)

            if eoc.order is not None:
                order = eoc.order
                min_h = np.exp(np.log(max_h) + np.log(min_e / max_e) / eoc.order)
                (line,) = ax.loglog(
                    [max_h, min_h],
                    [max_e, min_e],
                    "k--",
                )

        if abscissa and line is not None:
            if olabel is None:
                hname = abscissa.strip("$")
                if order == 1:
                    olabel = rf"$\mathcal{{O}}({hname})$"
                else:
                    olabel = rf"$\mathcal{{O}}({hname}^{{{order:g}}})$"

            if olabel:
                line.set_label(olabel)

        # }}}

        # {{{ plot order

        # }}}

        ax.grid(visible=True, which="major", linestyle="-", alpha=0.75)
        ax.grid(visible=True, which="minor", linestyle="--", alpha=0.5)

        if abscissa:
            ax.set_xlabel(abscissa)

        if ylabel:
            ax.set_ylabel(ylabel)

        if enable_legend and (len(eocs) > 1 or (line and olabel)):
            ax.legend()


# }}}


# {{{ matplotlib helpers


def check_usetex(*, s: bool) -> bool:
    try:
        import matplotlib
    except ImportError:
        return False

    try:
        return bool(matplotlib.checkdep_usetex(s))  # type: ignore[attr-defined]
    except AttributeError:
        # NOTE: simplified version from matplotlib
        # https://github.com/matplotlib/matplotlib/blob/ec85e725b4b117d2729c9c4f720f31cf8739211f/lib/matplotlib/__init__.py#L439=L456

        import shutil

        if not shutil.which("tex"):
            return False

        if not shutil.which("dvipng"):
            return False

        if not shutil.which("gs"):  # noqa: SIM103
            return False

        return True


def set_recommended_matplotlib(
    *,
    use_tex: bool | None = None,
    dark: bool | None = None,
    savefig_format: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> None:
    """Set custom :mod:`matplotlib` parameters.

    These are mainly used in the tests and examples to provide a uniform style
    to the results using `SciencePlots <https://github.com/garrettj403/SciencePlots>`__.
    For other applications, it is recommended to use local settings (e.g. in
    `matplotlibrc`).

    :arg use_tex: if *True*, LaTeX labels are enabled. By default, this checks
        if LaTeX is available on the system and only enables it if possible.
    :arg dark: if *True*, a dark default theme is selected instead of the
        default light one (see the ``dark_background`` theme of the ``SciencePlots``
        package). If *None*, this takes its values from the ``PYCAPUTO_DARK``
        boolean environment variable.
    :arg savefig_format: the format used when saving figures. By default, this
        uses the ``PYCAPUTO_SAVEFIG`` environment variable and falls back to
        the :mod:`matplotlib` parameter ``savefig.format``.
    :arg overrides: a mapping of parameters to override the defaults. These
        can also be set separately after this function was called.
    """
    try:
        import matplotlib.pyplot as mp
    except ImportError:
        return

    if use_tex is None:
        use_tex = "GITHUB_REPOSITORY" not in os.environ and check_usetex(s=True)

    if not use_tex:
        logger.warning("'use_tex' is disabled on this system.")

    if dark is None:
        tmp = os.environ.get("PYCAPUTO_DARK", "off").lower()
        dark = BOOLEAN_STATES.get(tmp, False)

    if savefig_format is None:
        savefig_format = os.environ.get(
            "PYCAPUTO_SAVEFIG", mp.rcParams["savefig.format"]
        ).lower()

    defaults: dict[str, dict[str, Any]] = {
        "figure": {
            "figsize": (8, 8),
            "dpi": 300,
            "constrained_layout.use": True,
        },
        "savefig": {"format": savefig_format},
        "text": {"usetex": use_tex},
        "legend": {"fontsize": 20},
        "lines": {"linewidth": 2, "markersize": 10},
        "axes": {
            "labelsize": 28,
            "titlesize": 28,
            "grid": True,
            "grid.axis": "both",
            "grid.which": "both",
            # NOTE: preserve existing colors (the ones in "science" are ugly)
            "prop_cycle": mp.rcParams["axes.prop_cycle"],
        },
        "xtick": {"labelsize": 20, "direction": "inout"},
        "ytick": {"labelsize": 20, "direction": "inout"},
        "xtick.major": {"size": 6.5, "width": 1.5},
        "ytick.major": {"size": 6.5, "width": 1.5},
        "xtick.minor": {"size": 4.0},
        "ytick.minor": {"size": 4.0},
    }

    with suppress(ImportError):
        import scienceplots  # noqa: F401

    with suppress(ImportError):
        import SciencePlots  # noqa: F401

    if "science" in mp.style.library:
        if dark:
            mp.style.use(["science", "ieee", "dark_background"])
        else:
            mp.style.use(["science", "ieee"])
    elif "seaborn-v0_8" in mp.style.library:
        # NOTE: matplotlib v3.6 deprecated all the seaborn styles
        mp.style.use("seaborn-v0_8-dark" if dark else "seaborn-v0_8-white")
    elif "seaborn" in mp.style.library:
        # NOTE: for older versions of matplotlib
        mp.style.use("seaborn-dark" if dark else "seaborn-white")

    for group, params in defaults.items():
        mp.rc(group, **params)

    if overrides:
        for group, params in overrides.items():
            mp.rc(group, **params)


@contextmanager
def figure(
    filename: PathLike | None = None,
    nrows: int = 1,
    ncols: int = 1,
    *,
    projection: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> Iterator[Any]:
    """A small wrapper context manager around :class:`matplotlib.figure.Figure`.

    :arg nrows: number of rows of subplots.
    :arg ncols: number of columns of subplots.
    :arg projection: a projection for all the axes in this figure, see
        :mod:`matplotlib.projections`.
    :arg figsize: the size of the resulting figure, set to
        ``(L * ncols, L * nrows)`` by default.
    :arg kwargs: Additional arguments passed to :func:`savefig`.
    :returns: the :class:`~matplotlib.figure.Figure` that was constructed. On exit
        from the context manager, the figure is saved to *filename* and closed.
    """
    import matplotlib.pyplot as mp

    fig = mp.figure()
    for i in range(nrows * ncols):
        fig.add_subplot(nrows, ncols, i + 1, projection=projection)

    # FIXME: get size of one figure
    if figsize is None:
        figsize = (8.0 * ncols, 8.0 * nrows)
    fig.set_size_inches(*figsize)

    try:
        yield fig
    finally:
        if filename is not None:
            savefig(fig, filename, **kwargs)
        else:
            mp.show()  # type: ignore[no-untyped-call,unused-ignore]

        mp.close(fig)


def savefig(
    fig: Any, filename: PathLike, *, overwrite: bool = True, **kwargs: Any
) -> None:
    """A wrapper around :meth:`~matplotlib.figure.Figure.savefig`.

    :arg filename: a file name where to save the figure. If the file name does
        not have an extension, the default format from ``savefig.format`` is
        used.
    :arg overwrite: if *True*, any existing files are overwritten.
    """
    import matplotlib.pyplot as mp

    filename = pathlib.Path(filename)
    if not filename.suffix:
        ext = mp.rcParams["savefig.format"]
        filename = filename.with_suffix(f".{ext}").resolve()

    if not overwrite and filename.exists():
        raise FileExistsError(f"Output file '{filename}' already exists")

    logger.info("Saving '%s'", filename)
    fig.savefig(filename, **kwargs)


# }}}


# {{{ timing


@dataclass(frozen=True)
class TimingResult:
    """Statistics for a set of runs (see :func:`timeit`)."""

    walltime: float
    """Minimum walltime for a set of runs."""
    mean: float
    """Mean walltime for a set of runs."""
    std: float
    """Standard derivation for a set of runs."""

    @classmethod
    def from_results(cls, results: list[float]) -> TimingResult:
        """Gather statistics from a set of runs."""
        rs = np.array(results)

        return TimingResult(
            walltime=np.min(rs),
            mean=np.mean(rs),
            std=np.std(rs, ddof=1),
        )

    def __str__(self) -> str:
        return f"{self.mean:.5f}s ± {self.std:.3f}"


def timeit(
    stmt: Callable[[], Any],
    *,
    repeat: int = 32,
    number: int = 1,
    skip: int = 0,
) -> TimingResult:
    """Run *stmt* using :func:`timeit.repeat`.

    :arg repeat: number of times to call :func:`timeit.timeit` (inside of
        :func:`timeit.repeat`).
    :arg number: number of times to run the *stmt* in each call to
        :func:`timeit.timeit`.
    :arg skip: number of leading calls from *repeat* to skip, e.g. to
        avoid measuring an initial cache hit.
    :returns: a :class:`TimingResult` with statistics about the runs.
    """

    import timeit as _timeit

    r = _timeit.repeat(stmt=stmt, repeat=repeat + 1, number=number)
    return TimingResult.from_results(r[skip:])


@dataclass
class BlockTimer:
    """A context manager for timing blocks of code.

    .. code:: python

        with BlockTimer("my-code-block") as bt:
            # do some code

        print(bt)
    """

    name: str = "block"
    """An identifier used to differentiate the timer."""

    t_wall_start: float = field(init=False)
    t_wall: float = field(init=False)
    """Total wall time (set after ``__exit__``), obtained from
    :func:`time.perf_counter`.
    """

    t_proc_start: float = field(init=False)
    t_proc: float = field(init=False)
    """Total process time (set after ``__exit__``), obtained from
    :func:`time.process_time`.
    """

    @property
    def t_cpu(self) -> float:
        """Total CPU time, obtained from ``t_proc / t_wall``."""
        return self.t_proc / self.t_wall

    def __enter__(self) -> BlockTimer:
        self.t_wall = self.t_proc = 0.0
        self.t_wall_start = time.perf_counter()
        self.t_proc_start = time.process_time()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.t_wall = time.perf_counter() - self.t_wall_start
        self.t_proc = time.process_time() - self.t_proc_start

    def __str__(self) -> str:
        return f"{self.name}: {self.t_wall:.3e}s wall, {self.t_cpu:.3f}x cpu"

    def pretty(self) -> str:
        # NOTE: this matches how MATLAB shows the time from `toc`.
        return f"[{self.name}] Elapsed time is {self.t_wall:.5f} seconds."


# }}}


# {{{ scipy wrappers


def gamma(x: Any) -> Array:
    """Wrapper around :data:`scipy.special.gamma`."""
    try:
        from scipy.special import gamma as _gamma

        return np.array(_gamma(x))
    except ImportError:
        return np.array(np.vectorize(math.gamma)(x))


# }}}


# {{{ others


def cached_on_first_arg(
    func: Callable[Concatenate[T, P], R],
) -> Callable[Concatenate[T, P], R]:
    """Cache function that stores the return values in the first argument.

    The values are stored in the instance and will be cleared once the instance
    is destroyed. To clear the cache sooner, use

    .. code:: python

        func.clear_cached(obj)

    :arg func: a function whose return values are cached in its first argument.
        This can be a simple function or a class method.
    """
    cache_dict_name = f"_cached_method_{func.__module__}{func.__name__}"

    def wrapper(obj: T, /, *args: P.args, **kwargs: P.kwargs) -> R:
        key = frozenset(kwargs.items()) | frozenset(args)

        try:
            d = getattr(obj, cache_dict_name)
        except AttributeError:
            # NOTE: 'cache_dict_name' could not be found, so we create it
            object.__setattr__(obj, cache_dict_name, {})
            d = getattr(obj, cache_dict_name)

        try:
            result: R = d[key]
        except KeyError:
            # NOTE: key could not be found in 'cache_dict_name'
            d[key] = result = func(obj, *args, **kwargs)

        return result

    def clear_cached(obj: T) -> None:
        object.__delattr__(obj, cache_dict_name)

    from functools import update_wrapper

    new_wrapper = update_wrapper(wrapper, func)
    new_wrapper.clear_cached = clear_cached  # type: ignore[attr-defined]

    return new_wrapper


def single_valued(
    iterable: Iterable[T],
    eq: Callable[[T, T], bool] | None = None,
) -> T:
    """Retrieve a single value from the *iterable*.

    This function will return the first value from the *iterable* and assert that
    all other values are equal to it in the sense of the *eq* predicate. Note
    that the check is not performed when optimizations are turned on (as it is
    an assert).

    :arg eq: an equality predicate on the elements of *iterable*, defaulting to
        :func:`operator.eq`.
    :returns: the first value of *iterable*.
    """
    if eq is None:
        import operator

        eq = operator.eq

    iterable = iter(iterable)
    try:
        first = next(iterable)
    except StopIteration:
        raise ValueError("Iterable is empty") from None

    assert all(eq(first, other) for other in iterable)
    return first


def is_single_valued(
    iterable: Iterable[T],
    eq: Callable[[T, T], bool] | None = None,
) -> bool:
    """This is a boolean version of :func:`single_valued`.

    The function will return *True* if all the elements in the iterable are equal,
    in the sense of the *eq* predicate, and *False* otherwise. Note that, unlike
    :func:`single_valued`, an empty iterable will return *True* to match the
    behaviour of :func:`all`.

    :arg eq: an equality predicate on the elements of *iterable*, defaulting to
        :func:`operator.eq`.
    """
    if eq is None:
        import operator

        eq = operator.eq

    iterable = iter(iterable)

    try:
        first = next(iterable)
    except StopIteration:
        return True

    return all(eq(first, other) for other in iterable)


# }}}


# {{{ dataclass


def dc_asdict(dc: DataclassInstance, *, init_only: bool = True) -> dict[str, Any]:
    """
    :returns: a shallow copy of the fields in the dataclass *dc*.
    """
    if not is_dataclass(dc):
        raise TypeError("input 'dc' is not a dataclass")

    return dict(dc_items(dc, init_only=init_only))


def dc_items(
    dc: DataclassInstance, *, init_only: bool = True
) -> Iterator[tuple[str, Any]]:
    """
    :arg init_only: if *False*, all the fields of the dataclass are returned,
        even those with ``init=False``.
    :returns: tuples of the form ``(field, value)`` with the fields from the
        dataclass.
    """
    if not is_dataclass(dc):
        raise TypeError("input 'dc' is not a dataclass")

    from dataclasses import fields

    for f in fields(dc):
        if not init_only or f.init:
            yield f.name, getattr(dc, f.name)


def dc_stringify(
    dc: DataclassInstance | NamedTuple | dict[str, Any],
    header: tuple[str, str] | None = None,
) -> str:
    """Stringify a dataclass, namedtuple or dictionary in a fancy way.

    :returns: a string containing two columns with the object attributes and
        values shown in a nice way.
    """
    if is_dataclass(dc):
        assert not isinstance(dc, type)
        fields = dc_asdict(dc)
    elif isinstance(dc, tuple) and hasattr(dc, "_asdict"):  # is_namedtuple
        fields = dc._asdict()
    elif isinstance(dc, dict):
        fields = dc
    else:
        raise TypeError(f"unrecognized type: '{type(dc).__name__}'")

    width = len(max(fields, key=len))
    fmt = f"{{:{width}}} : {{}}"

    def stringify(v: Any) -> str:
        sv = repr(v)
        if len(sv) > 128:
            sv = f"{type(v).__name__}<...>"

        return sv

    instance_attrs = sorted(
        {k: stringify(v) for k, v in fields.items() if k != "name"}.items()
    )

    header_attrs = []
    if header is None:
        if not isinstance(dc, dict):
            header_attrs.append(("class", type(dc).__name__))
        if "name" in fields:
            header_attrs.append(("name", fields["name"]))
        if not header_attrs:
            header_attrs.append(("attribute", "value"))
    else:
        header_attrs.append(header)
    header_attrs.append(("-" * width, "-" * width))

    return "\n".join([
        "\t{}".format("\n\t".join(fmt.format(k, v) for k, v in header_attrs)),
        "\t{}".format("\n\t".join(fmt.format(k, v) for k, v in instance_attrs)),
    ])


# }}}
