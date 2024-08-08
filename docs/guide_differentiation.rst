A New Differentiation Method
============================

Computing the fractional derivative of a function can be done in two ways. The
recommended method (with included magic) is calling :func:`pycaputo.diff` as

.. code:: python

    df = diff(f, p, alpha)

which will automatically select an appropriate method to use given the point set
``p`` and the order ``alpha`` (see also
:func:`~pycaputo.differentiation.guess_method_for_order`). To manually call a
specific method, use the lower level :func:`pycaputo.differentiation.diff` instead as

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
First, all subclasses must be a :func:`~dataclasses.dataclass` and implement the
abstract methods of the base class. For example,

.. literalinclude:: ../examples/guide-differentiation.py
    :language: python
    :lineno-match:
    :start-after: [class-definition-start]
    :end-before: [class-definition-end]

Then, we have defined a set of functions useful for working with a discrete
fractional operator approximations. As mentioned above, the main functionality
is provided by the :func:`~pycaputo.differentiation.diff` method. However, we
also have

* :func:`~pycaputo.differentiation.quadrature_weights`: a function that can provide
  the underlying quadrature weights for a given method. Note that not all methods
  can be efficiently expressed as a weighted sum with quadrature weights, so they
  should not implement this method. However, for those that can, this offers a
  pleasant way of reusing and analyzing the weights.

* :func:`~pycaputo.differentiation.differentiation_matrix`: a function that
  gathers all the quadrature weights into a matrix operator. By default, this
  function is implemented in terms of ``quadrature_weights``, but can be
  overwritten by more efficient implementations.

* :func:`~pycaputo.differentiation.diffs`: a function that computes the
  fractional operator at a given point on a grid. Note that not all methods can
  efficiently evaluate the fractional operator at a single point (e.g. global
  spectral methods), so they should not implement this functionality.

* :func:`~pycaputo.differentiation.diff`: a function that computes the fractional
  operator at a set of points. By default this is implemented in terms of the
  ``diffs`` function, but this doesn't always make sense. For example, if we want
  to compute the Caputo fractional derivative on a uniform grid, this can be
  evaluated more efficiently using the FFT.

In general, we require that the :func:`~pycaputo.differentiation.diff` function be
implemented for all available methods, while the remaining functions are optional.
These methods are all registered with the :func:`~functools.singledispatch` mechanism.
An example setup is provided below.

.. literalinclude:: ../examples/guide-differentiation.py
    :language: python
    :lineno-match:
    :start-after: [register-start]
    :end-before: [register-end]

The complete example can be found in
:download:`examples/guide-differentiation.py <../examples/guide-differentiation.py>`.
