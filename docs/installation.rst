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

When running on the CI, the dependencies are pinned to specific versions to avoid
unexpected failures due to new releases. The pinned versions can be found in
``.github/requirements-text.txt``.

justfile
--------

This project uses the `just <https://just.systems/man/en/>`__ command runner to
group a few smaller commands that are useful for development. You can see what
they are by running

.. code:: sh

    just --list

Using the command runner, you can install the project in *editable mode* (as above)
by running

.. code:: sh

    just develop

When developing, it is recommended to run the general formatting and linting
tools that are used by the project. This can be easily done using

.. code:: sh

   just format
   just lint

The ``justfile`` also defines all the commands that are run on the CI, so you can
run them locally to make sure everything works as expected.
