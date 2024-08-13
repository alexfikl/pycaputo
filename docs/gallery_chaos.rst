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

    This example uses the Caputo derivative and reproduces Figure 5.26 from [Petras2011]_.
    The complete setup (with parameters) can be found in
    :download:`examples/gallery/van-der-pol.py <../examples/gallery/van-der-pol.py>`.

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

    This example uses the Caputo derivative and reproduces Figure 5.29 from [Petras2011]_.
    The complete setup (with parameters) can be found in
    :download:`examples/gallery/duffing.py <../examples/gallery/duffing.py>`.
