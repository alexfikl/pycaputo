.. _sec-quadrature:

Quadrature
==========

Computing the fractional integral of a function can be done in two ways. The
recommended method (with included magic) is calling :func:`pycaputo.quad` as

.. code:: python

    qf = quad(f, p, alpha)

which will automatically select an appropriate method to use given the point set
``p`` and the order ``alpha`` (see also
:func:`~pycaputo.quadrature.guess_method_for_order`). To manually call a specific
method, use :func:`pycaputo.quadrature.quad` instead as

.. code:: python

    from pycaputo.quadrature import quad
    from pycaputo.quadrature.riemann_liouville import Trapezoidal

    m = Trapezoidal(alpha)
    qf = quad(m, f, p)

This requires more setup, but gives more control over the method used to
approximate the integral. The :func:`pycaputo.quadrature.quad` method
is based on the :func:`~functools.singledispatch` mechanism and can be easily
extended to support additional methods.

.. note::

    Note that for integration, the order :math:`\alpha` must be negative. This is
    because a single differential operator is used to define both the derivative
    and the integral.

Example
-------

We give here the skeleton for implementing a new custom
:class:`~pycaputo.quadrature.QuadratureMethod`. First, all subclasses must be a
:func:`~dataclasses.dataclass` as

.. literalinclude:: ../examples/example-custom-quad.py
    :lines: 14-24
    :language: python
    :linenos:

Then, we can implement the :func:`~pycaputo.quadrature.quad` method by
registering it with the :func:`~functools.singledispatch` mechanism as

.. literalinclude:: ../examples/example-custom-quad.py
    :lines: 27-34
    :language: python
    :linenos:

The complete example can be found in
:download:`examples/example-custom-quad.py <../examples/example-custom-quad.py>`.

API Reference
-------------

.. autofunction:: pycaputo.quad

.. automodule:: pycaputo.quadrature

Riemann-Liouville Integral
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pycaputo.quadrature.riemann_liouville
