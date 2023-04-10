Welcome
=======

.. toctree::
    :maxdepth: 2
    :hidden:

    derivative
    algorithms
    misc
    references

.. warning::

   This package is currently in development and very experimental (the API
   can and will change frequently for the forseeable future). For more mature
   libraries see `differint <https://github.com/differint/differint>`__ in
   Python or `FractionalDiffEq.jl <https://github.com/SciFracX/FractionalDiffEq.jl>`__
   in Julia.

This package provides tools to compute fractional order derivatives and
integrals.

To get an idea of the intended API an workflow of the library, we have here
a small example using the classical Caputo fractional order derivative,
as defined in [Li2020]_

.. math::

    D^\alpha_C[f](x) = \frac{1}{\Gamma(n - \alpha)} \int_a^x
        \frac{f^{(n)}(s)}{(x - s)^{\alpha + 1 - n}} \,\mathrm{d}s,

where :math:`n = \lceil \alpha \rceil` is the smallest integer larger than
:math:`\alpha`. In this example, we take :math:`\alpha \in (0, 1)`, so :math:`n = 1`,
and use a simple test function

.. math::

    f(x) = (1 + x)^3

with its Caputo fractional order derivative given by

.. math::

    D^\alpha_C[f](x) = \frac{3}{\Gamma(2 - \alpha)} x^{1 - \alpha}
        + \frac{6}{\Gamma(3 - \alpha)} x^{2 - \alpha}
        + \frac{6}{\Gamma(4 - \alpha)} x^{3 - \alpha}

They are simply defined as

.. literalinclude:: ../examples/caputo-derivative-l1.py
    :lines: 11-20
    :language: python
    :linenos:

To estimate the derivative, we use the classical L1 method (see Chapter 4.1 in
[Li2020]_ or Chapter 3 in [Karniadakis2019]_). The methods are set up as
follows

.. literalinclude:: ../examples/caputo-derivative-l1.py
    :lines: 23-26
    :language: python
    :linenos:

We can then set up a grid and evaluate the derivative at all points

.. literalinclude:: ../examples/caputo-derivative-l1.py
    :lines: 29-32
    :language: python
    :linenos:

The resulting approximation can be see below

.. image:: caputo-derivative-l1.png
    :width: 75%
    :align: center
    :alt: Approximation of the Caputo derivative usig the L1 method


The complete example can be found in
:download:`examples/caputo-derivative-l1.py <../examples/caputo-derivative-l1.py>`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
