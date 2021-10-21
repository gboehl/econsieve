
EconSieve - Transposed-Ensemble Kalman Filter (TEnKF) and Nonlinear Path-Adjusting Smoother (NPAS)
================================================================================================================


Apart from the smoother (`npas`) and TEnKF, I stole quite some of the code from these two projects:

.. code-block:: bash

   * https://github.com/rlabbe/filterpy
   * https://github.com/pykalman/pykalman


They deserve most of the merits. I just made everything look way more complicated. Sometimes ``filterpy`` was more efficient, sometimes ``pykalman``. Unfortunately the ``pykalman`` project is orphaned. I tweaked something here and there:

* treating numerical errors in the UKF covariance matrix by looking for the nearest positive semi-definite matrix
* eliminating identical sigma points (yields speedup assuming that evaluation of each point is costly)
* extracting functions from classes and compile them using the @njit flag (speedup)
* major cleanup

NPAS is built from scratch. I barely did any testing as a standalone filter and just always used it in combination with the 'pydsge', where it works very well.

Some very rudimentary documentation `can be found here <https://econsieve.readthedocs.io/en/latest/readme.html>`_.

Installation with ``pip``
-------------------------------------------------------

Be sure that you are on Python 3.x. Then it's as simple as:

.. code-block:: bash

   pip install econsieve


Installation of bleeding-edge version using ``git``
---------------------------------------------------

First install ``git``. Linux users just use their respective repos. 

Windows users probably use anaconda and can do

.. code-block:: bash

   conda install -c anaconda git

in the conda shell `as they kindly tell us here <https://anaconda.org/anaconda/git>`_. Otherwise you can probably `get it here <https://git-scm.com/download/win>`_.

Then you can simply do

.. code-block:: bash

   pip install git+https://github.com/gboehl/econsieve

If you run it and it complains about missing packages, please let me know so that I can update the `setup.py`!

Alternatively you can clone the repository and then from within the cloned folder run (Windows user from the Anaconda Prompt):

.. code-block:: bash

   pip install .


Updating
--------

The package is updated very frequently (find the history of latest commits `here <https://github.com/gboehl/econsieve/commits/master>`_). I hence recommend pulling and reinstalling whenever something is not working right. Run:

.. code-block:: bash

   pip install --upgrade econsieve
   
Citation
--------

**pydsge** is developed by Gregor Boehl to simulate, filter, and estimate DSGE models with the zero lower bound on nominal interest rates in various applications (see [gregorboehl.com](https://gregorboehl.com) for research papers using the package). Please cite it with:

.. code-block::

    @Software{boehl2020,
      Title  = {econsieve -- A collection of linear and nonlinear (Kalman) Filters for economic models},
      Author = {Gregor Boehl},
      Year   = {2020},
      Url    = {https://github.com/gboehl/econsieve},
    }

We appreciate citations for **pydsge** because it helps us to find out how people have
been using the package and it motivates further work.



References
----------

Boehl, Gregor (2020). `Efficient Solution, Filtering and Estimation of Models with OBCs <http://gregorboehl.com/live/obc_boehl.pdf>`_. *Unpublished Manuscript*

