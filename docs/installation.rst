Installation
============

``pycaputo`` is a pure Python project (for now), so standard methods will
just work™. The latest released version can be installed directly from PyPI

.. code:: sh

    python -m pip install pycaputo

Building from Source
--------------------

This library uses `hatchling <https://hatch.pypa.io/latest/>`__ for its
build system. You will need to install it beforehand for development purposes.
To build wheels and source distributions, use the ``build`` tool

.. code:: sh

    python -m build --sdist
    python -m build --wheel

This will create some files in a `dist` directory that you can use or distribute.

Development
-----------

For development, you'll want to create a virtual environment (might want to
also look into ``pipx`` or ``pipenv``). This can be achieved with

.. code:: sh

    python -m venv /path/to/my-venv
    source /path/to/mu-venv/activate

Then, you can install the project in *editable mode* as follows

.. code:: sh

    python -m pip install --verbose --no-build-isolation --editable .

This will allow running the tests directly, building the documentation, and all
sorts of other things useful when working on the library. This is all very
standard and explained in the `official Python packaging guides
<https://packaging.python.org/en/latest/>`__, which you should consult if any of
these terms are confusing.
