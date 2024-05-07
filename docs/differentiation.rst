.. _sec-differentiation:

Differentiation
===============

Computing the derivative of a function can be done in two ways. The recommended
method (with included magic) is calling :func:`pycaputo.diff` as

.. code:: python

    df = diff(f, p, alpha)

which will automatically select an appropriate method to use given the point set
``p`` and the order ``alpha`` (see also
:func:`pycaputo.differentiation.guess_method_for_order`). To manually call a
specific method, use :func:`pycaputo.differentiation.diff` instead as

.. code:: python

    from pycaputo.differentiation import diff
    from pycaputo.differentiation.caputo import L2C

    m = L2C(alpha)
    df = diff(m, f, p)

This requires more setup, but gives more control over the method used to
approximate the derivative. The :func:`pycaputo.differentiation.diff` method
is based on the :func:`~functools.singledispatch` mechanism and can be easily
extended to support additional methods.

Example
-------

We give here the skeleton for implementing a new custom
:class:`~pycaputo.differentiation.DerivativeMethod`.
First, all subclasses must be a :func:`~dataclasses.dataclass` as

.. literalinclude:: ../examples/example-custom-diff.py
    :lines: 14-24
    :language: python
    :linenos:

Then, we can implement the :func:`~pycaputo.differentiation.diff` method by
registering it with the :func:`~functools.singledispatch` mechanism as

.. literalinclude:: ../examples/example-custom-diff.py
    :lines: 27-34
    :language: python
    :linenos:

The complete example can be found in
:download:`examples/example-custom-diff.py <../examples/example-custom-diff.py>`.

API Reference
-------------

.. autofunction:: pycaputo.diff

.. automodule:: pycaputo.differentiation

Caputo Derivative
^^^^^^^^^^^^^^^^^

.. automodule:: pycaputo.differentiation.caputo

Riemann-Liouville Derivative
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pycaputo.differentiation.riemann_liouville
