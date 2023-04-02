# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from typing import Any, Iterable, List, Optional, Protocol, Tuple, TypeVar

import numpy as np

#: A generic invariant :class:`typing.TypeVar`.
T = TypeVar("T")


class ScalarFunction(Protocol):
    """A generic callable that can be evaluated at :math:`x`."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        :arg x: a scalar or array at which to evaluate the function.
        """
        ...


# {{{ Estimated Order of Convergence (EOC)


class EOCRecorder:
    """Keep track of all your *estimated order of convergence* needs.

    .. attribute:: estimated_order

        Estimated order of convergence for currently available data. The
        order is estimated by least squares through the given data
        (see :func:`estimate_order_of_convergence`).

    .. attribute:: max_error

        Largest error (in absolute value) in current data.

    .. automethod:: __init__
    .. automethod:: add_data_point
    .. automethod:: satisfied
    """

    def __init__(self, *, name: str = "Error", dtype: Any = None) -> None:
        if dtype is None:
            dtype = np.float64
        dtype = np.dtype(dtype)

        self.name = name
        self.dtype = dtype
        self.history: List[Tuple[np.ndarray, np.ndarray]] = []

    @property
    def _history(self) -> np.ndarray:
        return np.array(self.history, dtype=self.dtype).T

    def add_data_point(self, h: Any, error: Any) -> None:
        """
        :arg h: abscissa, a value representative of the "grid size".
        :arg error: error at given *h*.
        """
        self.history.append(
            (np.array(h, dtype=self.dtype), np.array(error, dtype=self.dtype))
        )

    @property
    def estimated_order(self) -> float:
        import numpy as np

        if not self.history:
            return np.array(np.nan, dtype=self.dtype)

        h, error = self._history
        _, eoc = estimate_order_of_convergence(h, error)
        return eoc

    @property
    def max_error(self) -> float:
        return np.amax(
            np.array([error for _, error in self.history], dtype=self.dtype),
            initial=np.array(0.0, dtype=self.dtype),
        )

    def __str__(self) -> str:
        return stringify_eoc(self)


def estimate_order_of_convergence(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Computes an estimate of the order of convergence in the least-square sense.
    This assumes that the :math:`(x, y)` pair follows a law of the form

    .. math::

        y = m x^p

    and estimates the constant :math:`m` and power :math:`p`.
    """
    assert x.size == y.size
    if x.size <= 1:
        raise RuntimeError("Need at least two values to estimate order.")

    eps = np.finfo(x.dtype).eps  # type: ignore[no-untyped-call]
    c = np.polyfit(np.log10(x + eps), np.log10(y + eps), 1)
    return 10 ** c[-1], c[-2]


def estimate_gliding_order_of_convergence(
    x: np.ndarray, y: np.ndarray, *, gliding_mean: Optional[int] = None
) -> np.ndarray:
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


def flatten(iterable: Iterable[Iterable[T]]) -> Tuple[T, ...]:
    from itertools import chain

    return tuple(chain.from_iterable(iterable))


def stringify_eoc(*eocs: EOCRecorder) -> str:
    r"""
    :arg eocs: an iterable of :class:`EOCRecorder`\ s that are assumed to have
        the same number of entries in their histories.
    :returns: a string representing the results in *eocs* in the
        GitHub Markdown format.
    """
    histories = [eoc._history for eoc in eocs]  # pylint: disable=protected-access
    orders = [
        estimate_gliding_order_of_convergence(h, error, gliding_mean=2)
        for h, error in histories
    ]

    h = histories[0][0]
    ncolumns = 1 + 2 * len(eocs)
    nrows = h.size

    lines = []
    lines.append(("h",) + flatten([(eoc.name, "EOC") for eoc in eocs]))

    lines.append((":-:",) * ncolumns)

    for i in range(nrows):
        lines.append(
            (f"{h[i]:.3e}",)
            + flatten(
                [
                    (f"{error[i]:.6e}", "---" if i == 0 else f"{order[i - 1, i]:.3f}")
                    for (_, error), order in zip(histories, orders)
                ]
            )
        )

    lines.append(
        ("Overall",) + flatten([("", f"{eoc.estimated_order:.3f}") for eoc in eocs])
    )

    widths = [max(len(line[i]) for line in lines) for i in range(ncolumns)]
    formats = ["{:%s}" % w for w in widths]

    return "\n".join(
        [
            " | ".join(fmt.format(value) for fmt, value in zip(formats, line))
            for line in lines
        ]
    )


# }}}
