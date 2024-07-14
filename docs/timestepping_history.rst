History (Checkpointing)
=======================

As is well-known, fractional-order derivatives use the whole history of a process.
The :class:`~pycaputo.history.History` class is meant to provide a way to store
and load items from the history in a uniform way, independent on the way the
data is actually stored. For example, the whole history can reside in memory
or it can reside on disk and be loaded only when necessary.

In general, the data that is stored depends on the type of equation that is being
solve and the numerical method used to solve it. However, the usage will always be

.. code:: python

    # load a value
    t, f = history[k]
    # .. perform some computations with the stored data ...
    ynext = y + dt ** alpha * f
    # ... store the new value ...
    history.append(t + dt, ynext)

Users that are just trying to obtain the solution of an FODE will not have to
interact with this functionality. The :func:`~pycaputo.stepping.evolve` function
returns appropriate events that contain the solution and the users are responsible
for storage.

.. automodule:: pycaputo.history
