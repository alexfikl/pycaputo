# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""This example is a little experiment on using numba to speed up some code.

Ideally we could just wrap a function up in :unfc:`numba.njit` and use it for
amazing speedups, but that of course does not work. This examples shows how the
:class:`pycaputo.differentiation.caputo.L1` method can be modified to work with
numba.

While this works, the current performance improvements seem to be quite modest.
However, this gives an idea of the work that would need to be done to allow
the code to work with numba. The ``numba-scipy`` library is an example of how
to wrap this into a more generic interface.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pycaputo.differentiation import caputo, diff
from pycaputo.grid import make_uniform_points
from pycaputo.typing import Array
from pycaputo.utils import timeit

try:
    import numba
    from numba.experimental import jitclass
    from numba.extending import overload
except ImportError:
    print("ERROR: this example requires the 'numba' package")
    raise SystemExit(0) from None


# {{{ redefine with acceleration

# Numba does not work with our dataclasses directly, although they're generally
# pretty simple. Therefore, we first use `jitclass` to define a little class that
# can be used by numba without changes.


@jitclass([("alpha", numba.float64)])
class NumbaL1:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha


# This is also the case for some additional classes that are passed to `diff`.
# In the L1 method, we just need ``x`` and ``dx``, so we create a small wrapper
# class for the ``Points`` dataclass as well.


@jitclass([("x", numba.float64[:]), ("dx", numba.float64[:])])
class NumbaPoints:
    def __init__(self, x: Array, dx: Array) -> None:
        self.x = x
        self.dx = dx


# Next, numba does not seem to recognize any inner function calls, so we need
# to teach it about our private functions that get reused in the main call.
# Luckily, this is very straightforward if the functions themselves are already
# numba compatible.

# NOTE: the type annotations seem to be necessary here and they need to match
# those of the original functions that are getting wrapped.


@overload(caputo._caputo_l1_weights)  # type: ignore[misc]
def _(x: Array, dx: Array, n: int, alpha: float) -> Callable[..., Array]:
    return caputo._caputo_l1_weights


@overload(caputo._caputo_piecewise_constant_integral)  # type: ignore[misc]
def _(x: Array, alpha: float) -> Callable[..., Array]:
    return caputo._caputo_piecewise_constant_integral


# Finally, we compile the main L1 diff implementation by looking it up in the
# ``diff.registry`` mapping. We use ``njit`` to hopefully get the maximum
# amount of performance out of this!

# NOTE: This is copy-pasted here because the original implementation has a
# call to the unsupported `callable` function and tries to use the input as a
# callable. This breaks the type signature for numba.

# FIXME: Is there some way to make numba work this this? Maybe we could just be
# stricter in our interface, but many methods expect a callable too..


@numba.jit(nopython=True)  # type: ignore[misc]
def diff_fast(m: NumbaL1, f: Array, p: NumbaPoints) -> Array:
    alpha = m.alpha
    x = p.x
    dx = p.dx

    # FIXME: in the uniform case, we can also do an FFT, but we need different
    # weights for that, so we leave it like this for now
    df = np.empty_like(fx)
    df[0] = np.nan

    for n in range(1, df.size):
        w = caputo._caputo_l1_weights(x, dx, n, alpha)
        df[n] = np.sum(w * fx[: w.size])

    return df


# }}}


# {{{ evaluate


def f(x: Array) -> Array:
    return (0.5 - x) ** 4


# setup
alpha = 0.9
l1_basic = caputo.L1(alpha=alpha)
l1_numba = NumbaL1(alpha=alpha)

# evaluate
p = make_uniform_points(3 * 1024, a=0.0, b=1.0)
p_numba = NumbaPoints(p.x, p.dx)
fx = f(p.x)

diff_num_basic = diff(l1_basic, fx, p)
diff_num_numba = diff_fast(l1_numba, fx, p_numba)

print(diff_fast.inspect_types())

result = timeit(lambda: diff(l1_basic, fx, p), skip=3)
print(f"Basic: {result}")
result = timeit(lambda: diff_fast(l1_numba, fx, p_numba), skip=3)
print(f"Numba: {result}")

# }}}
