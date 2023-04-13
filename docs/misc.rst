Utilities
=========

Grids
-----

.. automodule:: pycaputo.grid

Special Functions
-----------------

This module contains some analytical expressions for known fractional order
derivatives. The classes are meant to be used as

.. code-block:: python

    func = CaputoPolynomialDerivative(
        order=0.9,
        side=Side.Left,
        a=((1.0, 3.0), (2.0, 5.0)),
        )

    print("Function at 0.5: ", func(0.5))
    print("Derivative at 0.5: ", func.diff(0.5))

.. automodule:: pycaputo.functions

Logging
-------

.. automodule:: pycaputo.logging

Misc
----

.. automodule:: pycaputo.utils
