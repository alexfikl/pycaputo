Neural Models Gallery
=====================

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
    :download:`examples/gallery/adex.py <../examples/gallery/adex.py>`.

FitzHugh-Nagumo Models
----------------------

.. card:: Fractional-order FitzHugh-Nagumo System
    :class-title: sd-text-center

    .. image:: _static/gallery-fitzhugh-nagumo-light.svg
        :class: only-light
        :width: 65%
        :align: center
        :alt: Fractional-order FitzHugh-Nagumo system

    .. image:: _static/gallery-fitzhugh-nagumo-dark.svg
        :class: only-dark
        :width: 65%
        :align: center
        :alt: Fractional-order FitzHugh-Nagumo system

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.FitzHughRinzel` class to reproduce Figure 4d
    from [Brandibur2018]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/fitzhugh-nagumo.py <../examples/gallery/fitzhugh-nagumo.py>`.

FitzHugh-Rinzel Models
----------------------

.. card:: Fractional-order FitzHugh-Rinzel System
    :class-title: sd-text-center

    .. image:: _static/gallery-fitzhugh-rinzel-light.svg
        :class: only-light
        :width: 65%
        :align: center
        :alt: Fractional-order FitzHugh-Rinzel system

    .. image:: _static/gallery-fitzhugh-rinzel-dark.svg
        :class: only-dark
        :width: 65%
        :align: center
        :alt: Fractional-order FitzHugh-Rinzel system

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.FitzHughRinzel` class to reproduce Figure 3g
    from [Mondal2019]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/fitzhugh-rinzel.py <../examples/gallery/fitzhugh-rinzel.py>`.

Morris-Lecar Models
-------------------

.. card:: Fractional-order Morris-Lecar System
    :class-title: sd-text-center

    .. image:: _static/gallery-morris-lecar-light.svg
        :class: only-light
        :width: 65%
        :align: center
        :alt: Fractional-order Morris-Lecar system

    .. image:: _static/gallery-morris-lecar-dark.svg
        :class: only-dark
        :width: 65%
        :align: center
        :alt: Fractional-order Morris-Lecar system

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.MorrisLecar` class to reproduce Figure 11
    from [Shi2014]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/morris-lecar.py <../examples/gallery/morris-lecar.py>`.

Hindmarsh-Rose Models
---------------------

.. card:: Fractional-order two-dimensional Hindmarsh-Rose System
    :class-title: sd-text-center

    .. image:: _static/gallery-hindmarsh-rose2-light.svg
        :class: only-light
        :width: 65%
        :align: center
        :alt: Fractional-order two-dimensional Hindmarsh-Rose system

    .. image:: _static/gallery-hindmarsh-rose2-dark.svg
        :class: only-dark
        :width: 65%
        :align: center
        :alt: Fractional-order two-dimensional Hindmarsh-Rose system

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.HindmarshRose2` class to reproduce Figure 3b
    from [Kaslik2017]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/hindmarsh-rose2.py <../examples/gallery/hindmarsh-rose2.py>`.

.. card:: Fractional-order three-dimensional Hindmarsh-Rose System
    :class-title: sd-text-center

    .. image:: _static/gallery-hindmarsh-rose3-light.svg
        :class: only-light
        :width: 65%
        :align: center
        :alt: Fractional-order three-dimensional Hindmarsh-Rose system

    .. image:: _static/gallery-hindmarsh-rose3-dark.svg
        :class: only-dark
        :width: 65%
        :align: center
        :alt: Fractional-order three-dimensional Hindmarsh-Rose system

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.HindmarshRose3` class to reproduce Figure 5a
    from [Kaslik2017]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/hindmarsh-rose3.py <../examples/gallery/hindmarsh-rose3.py>`.

.. card:: Fractional-order four-dimensional Hindmarsh-Rose System
    :class-title: sd-text-center

    .. image:: _static/gallery-hindmarsh-rose4-light.svg
        :class: only-light
        :width: 65%
        :align: center
        :alt: Fractional-order four-dimensional Hindmarsh-Rose system

    .. image:: _static/gallery-hindmarsh-rose4-dark.svg
        :class: only-dark
        :width: 65%
        :align: center
        :alt: Fractional-order four-dimensional Hindmarsh-Rose system

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.HindmarshRose4` class to reproduce Figure 1
    from [Giresse2019]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/hindmarsh-rose4.py <../examples/gallery/hindmarsh-rose4.py>`.

Hodgkin-Huxley Models
---------------------

.. card:: Fractional-order Hodgkin-Huxley System
    :class-title: sd-text-center

    .. image:: _static/gallery-hodgkin-huxley-light.svg
        :class: only-light
        :width: 65%
        :align: center
        :alt: Fractional-order Hodgkin-Huxley system

    .. image:: _static/gallery-hodgkin-huxley-dark.svg
        :class: only-dark
        :width: 65%
        :align: center
        :alt: Fractional-order Hodgkin-Huxley system

    +++

    This example uses the Caputo derivative and the
    :class:`~pycaputo.fode.gallery.HodgkinHuxley` class to reproduce Figure 4
    from [Nagy2014]_. The complete setup (with parameters) can be found in
    :download:`examples/gallery/hodgkin-huxley.py <../examples/gallery/hodgkin-huxley.py>`.
