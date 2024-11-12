Neural Model Gallery
====================

.. note::

   This gallery contains examples of single neuron models. If you have any other
   favourite models that you'd like to add to this, they're very welcome!

   Some similar (integer order) examples can be found in exercises from the
   `Neuronal Dynamics <https://neuronaldynamics-exercises.readthedocs.io/en/latest/exercises/index.html>`__
   book by Gerstner, Kistler, Naud, and Paninski. The fractional ones will behave
   slightly differently, but should be comparable.

Integrate-and-Fire Models
-------------------------

.. card:: Fractional-order Perfect Integrate-and-Fire (PIF) System
    :class-title: sd-text-center

    .. image:: _static/gallery-pif-light.svg
        :class: only-light
        :width: 65%
        :align: center
        :alt: Fractional-order perfect integrate-and-fire system

    .. image:: _static/gallery-pif-dark.svg
        :class: only-dark
        :width: 65%
        :align: center
        :alt: Fractional-order perfect integrate-and-fire system

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.integrate_fire.pif.PIFModel` class. The complete setup
    (with parameters) can be found in
    :download:`examples/gallery/pif.py <../examples/gallery/pif.py>`.

.. card:: Fractional-order Leaky Integrate-and-Fire (LIF) System
    :class-title: sd-text-center

    .. image:: _static/gallery-lif-light.svg
        :class: only-light
        :width: 65%
        :align: center
        :alt: Fractional-order leaky integrate-and-fire system

    .. image:: _static/gallery-lif-dark.svg
        :class: only-dark
        :width: 65%
        :align: center
        :alt: Fractional-order leaky integrate-and-fire system

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.integrate_fire.lif.LIFModel` class. The complete setup
    (with parameters) can be found in
    :download:`examples/gallery/lif.py <../examples/gallery/lif.py>`.

.. card:: Fractional-order Adaptive Exponential Integrate-and-Fire (AdEx) System
    :class-title: sd-text-center

    .. image:: _static/gallery-ad-ex-light.svg
        :class: only-light
        :width: 65%
        :align: center
        :alt: Fractional-order adaptive exponential integrate-and-fire system

    .. image:: _static/gallery-ad-ex-dark.svg
        :class: only-dark
        :width: 65%
        :align: center
        :alt: Fractional-order adaptive exponential integrate-and-fire system

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.integrate_fire.ad_ex.AdExModel` class to reproduce Figure4d
    from [Naud2008]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/lif.py <../examples/gallery/lif.py>`.
