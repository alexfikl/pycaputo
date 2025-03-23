# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

"""This example is a little experiment on using jax to speed up some code.

Jax relies on tracing a function and then JIT-ing the result, so it works pretty
well with our code. However, code has to be specially written to work with jax:
* No in-place modification is allowed: code should prefer to concatenate
  or use ``np.where`` equivalents to work around that.
* Using ``array_api_compat``: we need to retrieve the appropriate backend when
  creating and working with arrays. Not all functions are modified to use this.

For this example, we have updated the L1 method to work with Jax. This seems to
give very impressive performance results. The output from running the example
on a recent machine is::

    Error 1.227745625165e-13
    Basic: 0.14187s ± 0.006
    Jax: 0.00188s ± 0.004

However, the compilation times are rather ridiculous, getting to around 2-3min.
"""

from __future__ import annotations

from pycaputo.differentiation import caputo, diff
from pycaputo.grid import Points, make_uniform_points
from pycaputo.typing import Array
from pycaputo.utils import BlockTimer, timeit

# {{{ jax

try:
    import jax
    import jax.numpy as jnp
    from jax.tree_util import register_dataclass

    jax.config.update("jax_enable_x64", val=True)  # type: ignore[no-untyped-call,unused-ignore]
except ImportError:
    print("ERROR: this example requires the 'jax' package")
    raise SystemExit(0) from None

# To make our code work with Jax, we need to register the dataclass as a PyTree
# so that it can be passed around in the functions.
#
register_dataclass(Points)

# We also need to access the underlying L1 method implementation from ``diff.registry``
# to avoid jitting the entire ``singledispatch`` implementation. With these two
# changes, everything seems to work as expected!

diff_jax = jax.jit(diff.registry[caputo.L1], static_argnums=(0,))


def f(x: Array) -> Array:
    return (0.5 - x) ** 4


# }}}

# {{{ evaluate

# setup
alpha = 0.9
method = caputo.L1(alpha=alpha)

# evaluate using numpy
p = make_uniform_points(3 * 1024, a=0.0, b=1.0)
fx = f(p.x)
diff_num_basic = diff(method, fx, p)

# evaluate using jax
p_jax = Points(p.a, p.b, x=jnp.array(p.x))  # type: ignore[arg-type,unused-ignore]
fx_jax = f(p_jax.x)
with BlockTimer("jax-compilation") as bt:
    diff_num_jax = diff_jax(method, fx, p_jax)
print(f"Initial jax JIT compilation time: {bt}")

diff_error = jnp.linalg.norm(diff_num_basic[1:] - diff_num_jax[1:])
print(f"Error {diff_error:.12e}")

# timing
result = timeit(lambda: diff(method, fx, p), skip=3)
print(f"Basic: {result}")
result = timeit(lambda: diff_jax(method, fx_jax, p_jax), skip=3)
print(f"Jax: {result}")

# }}}
