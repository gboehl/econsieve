EconSieve - Kalman Filter, Unscented Kalman filter, Ensemble Filter and Iterative Path-Adjusting Smoother (IPAS) 
---------------------------------------------------------------------------------------------

Apart from the smoother, I literally stole most of the code from these two projects:

    * https://github.com/rlabbe/filterpy
    * https://github.com/pykalman/pykalman

They deserve most of the merits. I just made everything look way more complicated. Sometimes `filterpy` was more efficient, sometimes `pykalman`. Unfortunately the `pykalman` project is orphaned. I tweaked something here and there:

   * treating numerical errors in the UKF covariance matrix by looking for the nearest positive semi-definite matrix
   * eliminating identical sigma points (yields speedup assuming that evaluation of each point is costly)
   * extracting functions from classes and compile them using the @njit flag (speedup)
   * major cleanup

IPAS is build from scratch. I barely did any testing as a standalone filter but always used it in combination with the 'pydsge' API, where it works very well.

Yet I have not updated the documentation or the licensing.

Also, basic functionallity is missing like f.i. a setup script.
