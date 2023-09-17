.. _sec-quadrature:

Quadrature
==========

Computing the fractional integral of a function can be done in two ways. The
recommended method (with included magic) is calling :func:`pycaputo.quad` as

.. code:: python

   df = quad(f, p, alpha)

which will automatically select an appropriate method to use given the point set
``p`` and the order ``alpha``. To manually select a method use e.g. the
``method="RiemannLiouvilleTrapezoidalMethod"`` keyword. The lower level function
call is given by :func:`pycaputo.quadrature.quad` which can be used as
(with a negative order ``alpha``)

.. code:: python

   d = RiemannLiouvilleDerivative(order=alpha, side=Side.Left)
   m = RiemannLiouvilleTrapezoidalMethod(d)
   qf = quad(m, f, p)

This requires more setup, but gives more control over the method used to
approximate the integral. The :func:`pycaputo.quadrature.quad` method
is based on the :func:`~functools.singledispatch` mechanism and can be easily
extended to support additional methods.

Example
-------

We give here the skeleton for implementing a new custom
:class:`~pycaputo.quadrature.QuadratureMethod`.
First, all subclasses must be a :func:`~dataclasses.dataclass` as

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
