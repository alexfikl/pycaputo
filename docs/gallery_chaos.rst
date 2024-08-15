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

    This example uses the Caputo derivative. The complete setup (with parameters)
    can be found in
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

    This example uses the Caputo derivative and reproduces Figure 5.29 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/duffing.py <../examples/gallery/duffing.py>`.

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

    This example uses the Caputo derivative and reproduces Figure 5.26 from
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

    This example uses the Caputo derivative and reproduces Figure 5.43 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/arneodo.py <../examples/gallery/arneodo.py>`.

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

    This example uses the Caputo derivative and reproduces Figure 5.32 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/lorenz.py <../examples/gallery/lorenz.py>`.


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

    This example uses the Caputo derivative and reproduces Figure 5.33 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/chen.py <../examples/gallery/chen.py>`.

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

    This example uses the Caputo derivative and reproduces Figure 5.40 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/genesio_tesi.py <../examples/gallery/genesio_tesi.py>`.

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

    This example uses the Caputo derivative and reproduces Figure 5.35 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/lu.py <../examples/gallery/lu.py>`.

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

    This example uses the Caputo derivative and reproduces Figure 5.37 from
    [Petras2011]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/liu.py <../examples/gallery/liu.py>`.
