A New Fractional Operator
=========================

The first step for introducing new discretization methods into the ``pycaputo``
is defining a new fractional operator. A fractional operator is a subclass
of :class:`~pycaputo.derivatives.FractionalOperator` that defines its main
parameters.

For example, the Riemann--Liouville derivative is defined as follows in
:class:`~pycaputo.derivatives.RiemannLiouvilleDerivative`.

.. literalinclude:: ../src/pycaputo/derivatives.py
    :lines: 33-55
    :lineno-match:
    :language: python

As we can see, the Riemann-Liouville operator is defined by its order :math:`\alpha`
and by the side of the integration. The class itself is not meant to implement
any logic and is just used to hold the information about the operator itself.
Many places can then be generic with respect to the operator type, e.g.
:func:`~pycaputo.special.sin_derivative`.

To showcase how a new and more complex fractional operator can be added to the
library, we briefly implement the Prabhakar integral (see [Karniadakis2019]_).
Mathematically, the operator is defined as

.. math::

    D^\gamma_{\alpha, \beta, \mu}[f](x) = \int_a^x
        (x - s)^{\beta - 1}
        E^\gamma_{\alpha, \beta}(\mu (x - s)^\alpha)
        f(s) \,\mathrm{d}s

where :math:`E^\gamma_{\alpha, \beta}` is the three-parameter Mittag-Leffler
function (also known as the Prabhakar function). As we can see, for
:math:`\gamma = \mu = 0` and :math:`\alpha = \beta`, this reduces to the
previous Riemann-Liouville integral.

This operator has a total of 4 parameters and can be simply implemented as below.

.. literalinclude:: ../examples/example-custom-derivative.py
    :lines: 8-23
    :lineno-match:
    :language: python

For a complete list of already defined operators see :mod:`pycaputo.derivatives`.
