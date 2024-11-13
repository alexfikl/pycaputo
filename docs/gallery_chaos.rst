Chaotic Systems Gallery
=======================

.. note::

    This gallery is mainly inspired by Chapter 5 from [Petras2011]_. If you have any
    other favourite chaotic system that you'd like to add to this, they're very
    welcome!

    A similarly wonderful gallery is available for
    `FractionalDiffEq.jl <https://scifracx.org/FractionalDiffEq.jl/dev/ChaosGallery/>`__.

.. warning::

   Some of the results here do not match their references as well as they maybe
   should. This may be down to bugs or the differences in the methods used for
   the discretization. Also, lets not forget that these systems are chaotic.

   For the comparisons to Chapter 5 from [Petras2011]_, the author describes the
   numerical methods in Section 2.9. The method used in the numerical examples is
   a Grünwald-Letnikov-type discretization of the Caputo derivative with limited
   memory. In contrast, we use a the higher-order Predictor-Corrector method
   with full memory, which presumably gives better results.


Two Dimensional Systems
-----------------------

.. card:: Fractional-order Brusselator System
    :class-title: sd-text-center

    .. image:: _static/gallery-brusselator-light.svg
        :class: only-light
        :width: 65%
        :align: center
        :alt: Fractional-order Brusselator system phase diagram

    .. image:: _static/gallery-brusselator-dark.svg
        :class: only-dark
        :width: 65%
        :align: center
        :alt: Fractional-order Brusselator system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.Brusselator` class. The complete setup
    (with parameters) can be found in
    :download:`examples/gallery/brusselator.py <../examples/gallery/brusselator.py>`.

.. card:: Fractional-order Duffing System
    :class-title: sd-text-center

    .. image:: _static/gallery-duffing-light.svg
        :class: only-light
        :width: 65%
        :align: center
        :alt: Fractional-order Duffing system phase diagram

    .. image:: _static/gallery-duffing-dark.svg
        :class: only-dark
        :width: 65%
        :align: center
        :alt: Fractional-order Duffing system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.Duffing` class to reproduce Figure 5.29 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/duffing.py <../examples/gallery/duffing.py>`.

.. card:: Fractional-order Lotka-Volterra System
    :class-title: sd-text-center

    .. image:: _static/gallery-lotka-volterra2-light.svg
        :class: only-light
        :width: 65%
        :align: center
        :alt: Fractional-order Lotka-Volterra system phase diagram

    .. image:: _static/gallery-lotka-volterra2-dark.svg
        :class: only-dark
        :width: 65%
        :align: center
        :alt: Fractional-order Lotka-Volterra system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.LotkaVolterra2` class. The complete setup
    (with parameters) can be found in
    :download:`examples/gallery/lotka-volterra.py <../examples/gallery/lotka-volterra.py>`.

.. card:: Fractional-order van der Pol System
    :class-title: sd-text-center

    .. image:: _static/gallery-van-der-pol-light.svg
        :class: only-light
        :width: 65%
        :align: center
        :alt: Fractional-order van der Pol system phase diagram

    .. image:: _static/gallery-van-der-pol-dark.svg
        :class: only-dark
        :width: 65%
        :align: center
        :alt: Fractional-order van der Pol system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.VanDerPol` class to reproduce Figure 5.26 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/van-der-pol.py <../examples/gallery/van-der-pol.py>`.

Three Dimensional Systems
-------------------------

.. card:: Fractional-order Arneodo System
    :class-title: sd-text-center

    .. image:: _static/gallery-arneodo-light.svg
        :class: only-light
        :width: 75%
        :align: center
        :alt: Fractional-order Arneodo system phase diagram

    .. image:: _static/gallery-arneodo-dark.svg
        :class: only-dark
        :width: 75%
        :align: center
        :alt: Fractional-order Arneodo system phase diagram

    +++

    This example uses the Caputo derivative and
    :class:`~pycaputo.fode.gallery.Arneodo` class to reproduce Figure 5.43 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/arneodo.py <../examples/gallery/arneodo.py>`.

.. card:: Fractional-order Cellular Neural Network (3 cells) System
    :class-title: sd-text-center

    .. image:: _static/gallery-cnn-light.svg
        :class: only-light
        :width: 75%
        :align: center
        :alt: Fractional-order Cellular Neural Network system phase diagram

    .. image:: _static/gallery-cnn-dark.svg
        :class: only-dark
        :width: 75%
        :align: center
        :alt: Fractional-order Cellular Neural Network system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.CellularNeuralNetwork3` class to reproduce
    Figure 5.58 from [Petras2011]_. The complete setup (with parameters) can be
    found in :download:`examples/gallery/cnn.py <../examples/gallery/cnn.py>`.

.. card:: Fractional-order Chen System
    :class-title: sd-text-center

    .. image:: _static/gallery-chen-light.svg
        :class: only-light
        :width: 75%
        :align: center
        :alt: Fractional-order Chen system phase diagram

    .. image:: _static/gallery-chen-dark.svg
        :class: only-dark
        :width: 75%
        :align: center
        :alt: Fractional-order Chen system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.Chen` class to reproduce Figure 5.33 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/chen.py <../examples/gallery/chen.py>`.

.. card:: Fractional-order Chua System
    :class-title: sd-text-center

    .. image:: _static/gallery-chua-light.svg
        :class: only-light
        :width: 65%
        :align: center
        :alt: Fractional-order Chua system phase diagram

    .. image:: _static/gallery-chua-dark.svg
        :class: only-dark
        :width: 65%
        :align: center
        :alt: Fractional-order Chua system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.Chua` class to reproduce
    Figure 5.6 from [Petras2011]_. The complete setup (with parameters) can be
    found in :download:`examples/gallery/chua.py <../examples/gallery/chua.py>`.

.. card:: Fractional-order Genesio-Tesi System
    :class-title: sd-text-center

    .. image:: _static/gallery-genesio-tesi-light.svg
        :class: only-light
        :width: 75%
        :align: center
        :alt: Fractional-order Genesio-Tesi system phase diagram

    .. image:: _static/gallery-genesio-tesi-dark.svg
        :class: only-dark
        :width: 75%
        :align: center
        :alt: Fractional-order Genesio-Tesi system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.GenesioTesi` class to reproduce Figure 5.40 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/genesio-tesi.py <../examples/gallery/genesio-tesi.py>`.

.. card:: Fractional-order Labyrinth System
    :class-title: sd-text-center

    .. image:: _static/gallery-labyrinth-light.svg
        :class: only-light
        :width: 75%
        :align: center
        :alt: Fractional-order Labyrinth system phase diagram

    .. image:: _static/gallery-labyrinth-dark.svg
        :class: only-dark
        :width: 75%
        :align: center
        :alt: Fractional-order Labyrinth system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.Labyrinth` class. The complete setup
    (with parameters) can be found in
    :download:`examples/gallery/labyrinth.py <../examples/gallery/labyrinth.py>`.

.. card:: Fractional-order Liu System
    :class-title: sd-text-center

    .. image:: _static/gallery-liu-light.svg
        :class: only-light
        :width: 75%
        :align: center
        :alt: Fractional-order Liu system phase diagram

    .. image:: _static/gallery-liu-dark.svg
        :class: only-dark
        :width: 75%
        :align: center
        :alt: Fractional-order Liu system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.Liu` class to reproduce Figure 5.37 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/liu.py <../examples/gallery/liu.py>`.

.. card:: Fractional-order Lorenz System
    :class-title: sd-text-center

    .. image:: _static/gallery-lorenz-light.svg
        :class: only-light
        :width: 75%
        :align: center
        :alt: Fractional-order Lorenz system phase diagram

    .. image:: _static/gallery-lorenz-dark.svg
        :class: only-dark
        :width: 75%
        :align: center
        :alt: Fractional-order Lorenz system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.Lorenz` class to reproduce Figure 5.32 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/lorenz.py <../examples/gallery/lorenz.py>`.

.. card:: Fractional-order Lorenz-84 System
    :class-title: sd-text-center

    .. image:: _static/gallery-lorenz84-light.svg
        :class: only-light
        :width: 75%
        :align: center
        :alt: Fractional-order Lorenz-84 system phase diagram

    .. image:: _static/gallery-lorenz84-dark.svg
        :class: only-dark
        :width: 75%
        :align: center
        :alt: Fractional-order Lorenz-84 system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.Lorenz84`. The complete setup (with parameters)
    can be found in
    :download:`examples/gallery/lorenz84.py <../examples/gallery/lorenz84.py>`.

.. card:: Fractional-order Lotka-Volterra (3 equation) System
    :class-title: sd-text-center

    .. image:: _static/gallery-lotka-volterra3-light.svg
        :class: only-light
        :width: 75%
        :align: center
        :alt: Fractional-order Lotka-Volterra system phase diagram

    .. image:: _static/gallery-lotka-volterra3-dark.svg
        :class: only-dark
        :width: 75%
        :align: center
        :alt: Fractional-order Lotka-Volterra system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.LotkaVolterra3` class to reproduce Figure 5.53
    from [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/lotka-volterra3.py <../examples/gallery/lotka-volterra3.py>`.

.. card:: Fractional-order Lü System
    :class-title: sd-text-center

    .. image:: _static/gallery-lu-light.svg
        :class: only-light
        :width: 75%
        :align: center
        :alt: Fractional-order Lü system phase diagram

    .. image:: _static/gallery-lu-dark.svg
        :class: only-dark
        :width: 75%
        :align: center
        :alt: Fractional-order Lü system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.Lu` class to reproduce Figure 5.35 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/lu.py <../examples/gallery/lu.py>`.

.. card:: Fractional-order Ma-Chen Financial System
    :class-title: sd-text-center

    .. image:: _static/gallery-ma-chen-light.svg
        :class: only-light
        :width: 75%
        :align: center
        :alt: Fractional-order Ma-Chen Financial system phase diagram

    .. image:: _static/gallery-ma-chen-dark.svg
        :class: only-dark
        :width: 75%
        :align: center
        :alt: Fractional-order Ma-Chen Financial system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.MaChen` class to reproduce Figure 5.55 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/ma-chen.py <../examples/gallery/ma-chen.py>`.

.. card:: Fractional-order Newton-Leipnik System
    :class-title: sd-text-center

    .. image:: _static/gallery-newton-leipnik-light.svg
        :class: only-light
        :width: 75%
        :align: center
        :alt: Fractional-order Newton-Leipnik system phase diagram

    .. image:: _static/gallery-newton-leipnik-dark.svg
        :class: only-dark
        :width: 75%
        :align: center
        :alt: Fractional-order Newton-Leipnik system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.NewtonLeipnik` class to reproduce Figure 5.46
    from [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/newton-leipnik.py <../examples/gallery/newton-leipnik.py>`.

.. card:: Fractional-order Rössler System
    :class-title: sd-text-center

    .. image:: _static/gallery-rossler-light.svg
        :class: only-light
        :width: 75%
        :align: center
        :alt: Fractional-order Rössler system phase diagram

    .. image:: _static/gallery-rossler-dark.svg
        :class: only-dark
        :width: 75%
        :align: center
        :alt: Fractional-order Rössler system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.Rossler` class to reproduce Figure 5.44 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/rossler.py <../examples/gallery/rossler.py>`.

.. card:: Fractional-order Qi System
    :class-title: sd-text-center

    .. image:: _static/gallery-qi-light.svg
        :class: only-light
        :width: 75%
        :align: center
        :alt: Fractional-order Qi system phase diagram

    .. image:: _static/gallery-qi-dark.svg
        :class: only-dark
        :width: 75%
        :align: center
        :alt: Fractional-order Qi system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.Qi` class. The complete setup (with parameters)
    can be found in :download:`examples/gallery/qi.py <../examples/gallery/qi.py>`.

.. card:: Fractional-order Volta System
    :class-title: sd-text-center

    .. image:: _static/gallery-volta-light.svg
        :class: only-light
        :width: 75%
        :align: center
        :alt: Fractional-order Volta system phase diagram

    .. image:: _static/gallery-volta-dark.svg
        :class: only-dark
        :width: 75%
        :align: center
        :alt: Fractional-order Volta system phase diagram

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.Volta` class to reproduce Figure 5.62 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/volta.py <../examples/gallery/volta.py>`.
