# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, is_dataclass
from types import TracebackType
from typing import Any, Concatenate, Literal, NamedTuple

import numpy as np

from pycaputo.logging import get_logger
from pycaputo.typing import Array, DataclassInstance, P, PathLike, R, T

log = get_logger(__name__)


# {{{ environment


# fmt: off
BOOLEAN_STATES = {
    1: True, "1": True, "yes": True, "true": True, "on": True, "y": True,
    0: False, "0": False, "no": False, "false": False, "off": False, "n": False,
}
# fmt: on


def get_environ_bool(name: str) -> bool:
    value = os.environ.get(name)
    return BOOLEAN_STATES.get(value.lower(), False) if value else False


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
        for h_i, e_i in zip(h, error, strict=True):
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
            for (_, error), order in zip(histories, orders, strict=True)
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
        " | ".join(fmt.format(value) for fmt, value in zip(formats, line, strict=True))
        for line in lines
    ])


def visualize_eoc(
    filename: PathLike,
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
        for eoc, marker in zip(eocs, markers, strict=True):
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
        return bool(matplotlib.checkdep_usetex(s))  # type: ignore[attr-defined,unused-ignore]
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

    # start off by resetting the defaults
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)

    import os

    if use_tex is None:
        use_tex = "GITHUB_REPOSITORY" not in os.environ and check_usetex(s=True)

    if not use_tex:
        log.warning("'use_tex' is disabled on this system.")

    if dark is None:
        dark = get_environ_bool("PYCAPUTO_DARK")

    if savefig_format is None:
        savefig_format = os.environ.get(
            "PYCAPUTO_SAVEFIG", mp.rcParams["savefig.format"]
        ).lower()

    # NOTE: preserve existing colors (the ones in "science" are ugly)
    prop_cycle = mp.rcParams["axes.prop_cycle"]
    with suppress(ImportError):
        import scienceplots  # noqa: F401

        mp.style.use(["science", "ieee"])

    # NOTE: the 'petroff10' style is available for version >= 3.10.0 and changes
    # the 'prop_cycle' to the 10 colors that are more accessible
    if "petroff10" in mp.style.available:
        mp.style.use("petroff10")
        prop_cycle = mp.rcParams["axes.prop_cycle"]

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
            "prop_cycle": prop_cycle,
        },
        "xtick": {"labelsize": 20, "direction": "out"},
        "ytick": {"labelsize": 20, "direction": "out"},
        "xtick.major": {"size": 6.5, "width": 1.5},
        "ytick.major": {"size": 6.5, "width": 1.5},
        "xtick.minor": {"size": 4.0},
        "ytick.minor": {"size": 4.0},
    }

    if dark:
        # NOTE: this is the black color used by the sphinx-book theme
        black = "111111"
        gray = "28313D"
        defaults["text"].update({"color": "white"})
        defaults["axes"].update({
            "labelcolor": "white",
            "facecolor": gray,
            "edgecolor": "white",
        })
        defaults["xtick"].update({"color": "white"})
        defaults["ytick"].update({"color": "white"})
        defaults["figure"].update({"facecolor": black, "edgecolor": black})
        defaults["savefig"].update({"facecolor": black, "edgecolor": black})

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
    pane_fill: bool = False,
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
        width, height = mp.rcParams["figure.figsize"]
        figsize = (width * ncols, height * nrows)
    fig.set_size_inches(*figsize)

    if projection == "3d":
        from mpl_toolkits.mplot3d.axes3d import Axes3D

        for ax in fig.axes:
            assert isinstance(ax, Axes3D)
            ax.xaxis.pane.fill = pane_fill
            ax.yaxis.pane.fill = pane_fill
            ax.zaxis.pane.fill = pane_fill

    try:
        yield fig
    finally:
        if projection == "3d":
            for ax in fig.axes:
                assert isinstance(ax, Axes3D)
                ax.set_box_aspect((4, 4, 4), zoom=1.1)

        if filename is not None:
            savefig(fig, filename, **kwargs)
        else:
            mp.show(block=True)  # type: ignore[no-untyped-call,unused-ignore]

        mp.close(fig)


def savefig(
    fig: Any,
    filename: PathLike,
    *,
    bbox_inches: str = "tight",
    pad_inches: float = 0,
    normalize: bool = False,
    overwrite: bool = True,
    **kwargs: Any,
) -> None:
    """A wrapper around :meth:`~matplotlib.figure.Figure.savefig`.

    :arg filename: a file name where to save the figure. If the file name does
        not have an extension, the default format from ``savefig.format`` is
        used.
    :arg normalize: if *True*, use :func:`slugify` to normalize the file name.
        Note that this will slugify any extensions as well and replace them
        with the default extension.
    :arg overwrite: if *True*, any existing files are overwritten.
    :arg kwargs: renaming arguments are passed directly to ``savefig``.
    """
    import pathlib

    import matplotlib.pyplot as mp

    ext = mp.rcParams["savefig.format"]
    filename = pathlib.Path(filename)

    if normalize:
        # NOTE: slugify(name) will clubber any prefixes, so we special-case a
        # few of them here to help out the caller
        if filename.suffix in {".png", ".jpg", ".jpeg", ".pdf", ".eps", ".tiff"}:
            filename = filename.with_stem(slugify(filename.stem))
        else:
            filename = filename.with_name(slugify(filename.name)).with_suffix(f".{ext}")

    if not filename.suffix:
        filename = filename.with_suffix(f".{ext}").resolve()

    if not overwrite and filename.exists():
        raise FileExistsError(f"Output file '{filename}' already exists")

    bbox_extra_artists = []
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            bbox_extra_artists.append(legend)

    log.info("Saving '%s'", filename)
    fig.savefig(
        filename,
        bbox_extra_artists=tuple(bbox_extra_artists),
        bbox_inches="tight",
        **kwargs,
    )


def slugify(stem: str, separator: str = "_") -> str:
    """
    :returns: an ASCII slug representing *stem*, with all the unicode cleaned up
        and all non-standard separators replaced.
    """
    import re
    import unicodedata

    stem = unicodedata.normalize("NFKD", stem)
    stem = stem.encode("ascii", "ignore").decode().lower()
    stem = re.sub(r"[^a-z0-9]+", separator, stem)
    stem = re.sub(rf"[{separator}]+", separator, stem.strip(separator))

    return stem


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
class TicTocTimer:
    """A simple timer that tries to copy MATLAB's ``tic`` and ``toc`` functions.

    .. code:: python

        time = TicTocTimer()
        time.tic()

        # ... do some work ...

        elapsed = time.toc()
        print(time)
    """

    t_wall_start: float = field(default=0.0, init=False)
    t_wall: float = field(default=0.0, init=False)

    n_calls: int = field(default=0, init=False)
    t_avg: float = field(default=0.0, init=False)
    t_sqr: float = field(default=0.0, init=False)

    def tic(self) -> None:
        self.t_wall = 0.0
        self.t_wall_start = time.perf_counter()

    def toc(self) -> float:
        self.t_wall = time.perf_counter() - self.t_wall_start

        # statistics
        self.n_calls += 1

        delta0 = self.t_wall - self.t_avg
        self.t_avg += delta0 / self.n_calls
        delta1 = self.t_wall - self.t_avg
        self.t_sqr += delta0 * delta1

        return self.t_wall

    def __str__(self) -> str:
        # NOTE: this matches how MATLAB shows the time from `toc`.
        return f"Elapsed time is {self.t_wall:.5f} seconds."

    def stats(self) -> str:
        """Aggregate statistics across multiple calls to :meth:`toc`."""
        # NOTE: n_calls == 0 => toc was not called yet, so stddev is zero
        #       n_calls == 1 => only one call to toc, so the stddev is zero
        t_std = np.sqrt(self.t_sqr / (self.n_calls - 1)) if self.n_calls > 1 else 0.0

        return f"avg {self.t_avg:.3f}s ± {t_std:.3f}s"

    def short(self) -> str:
        """A shorter string for the last :meth:`tic`-:meth:`toc` cycle."""
        return f"wall {self.t_wall:.5f}s"


@dataclass
class BlockTimer:
    """A context manager for timing blocks of code.

    .. code:: python

        with BlockTimer("my-code-block") as bt:
            # ... do some work ...

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
        import datetime

        t_wall = datetime.timedelta(seconds=round(self.t_wall))
        return f"{self.name}: {t_wall} wall, {self.t_cpu:.3f}x cpu"

    def pretty(self) -> str:
        # NOTE: this matches how MATLAB shows the time from `toc`.
        return f"[{self.name}] Elapsed time is {self.t_wall:.5f} seconds."


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
    cache_dict_name = f"_cached_function_{func.__module__}{func.__name__}"

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
