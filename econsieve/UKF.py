# -*- coding: utf-8 -*-

"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import (absolute_import, division)

from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import eye, zeros, dot, isscalar, outer, empty
from .unscented_transform import unscented_transform as UT
from .stats import logpdf
from .common import pretty_str
from warnings import warn
from numba import njit


@njit(cache=True)
def cross_variance(Wc, x, z, sigmas_f, sigmas_h):
    """
    Compute cross variance of the state `x` and measurement `z`.
    """

    Pxz = zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
    N = sigmas_f.shape[0]
    for i in range(N):
        dx = sigmas_f[i] - x
        dz = sigmas_h[i] - z
        Pxz += Wc[i] * outer(dx, dz)
    return Pxz

@njit(cache=True)
def update(z, R, P, x, Wc, Wm, sigmas_f, sigmas_h):
    """
    Update the UKF with the given measurements. On return,
    x and P contain the new mean and covariance of the filter.

    Parameters
    ----------

    z : numpy.array of shape (dim_z)
        measurement vector

    R : numpy.array((dim_z, dim_z)), optional
        Measurement noise. 

    """

    # mean and covariance of prediction passed through unscented transform
    zp, S = UT(sigmas_h, Wm, Wc, R)

    # compute cross variance of the state and the measurements
    Pxz = cross_variance(Wc, x, zp, sigmas_f, sigmas_h)

    y = z - zp   # residual

    # print(S)
    # print(np.linalg.cond(S))
    # print(np.linalg.det(S))
    # SI = np.linalg.pinv(S, rcond=1e-2)
    SI = np.linalg.pinv(S)

    K = dot(Pxz, SI)        # Kalman gain

    # update Gaussian state estimate (x, P)
    x = x + K @ y
    # print(x)
    P = P - dot(K, dot(S, K.T))

    return x, P, S, y

class UnscentedKalmanFilter(object):
    r"""
    Implements the Scaled Unscented Kalman filter (UKF) as defined by
    Simon Julier in [1], using the formulation provided by Wan and Merle
    in [2]. This filter scales the sigma points to avoid strong nonlinearities.


    Parameters
    ----------

    dim_x : int
        Number of state variables for the filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.


    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.

        This is for convience, so everything is sized correctly on
        creation. If you are using multiple sensors the size of `z` can
        change based on the sensor. Just provide the appropriate hx function

    hx : function(x)
        Measurement function. Converts state vector x into a measurement
        vector of shape (dim_z).

    fx : function(x)
        function that returns the state x transformed by the
        state transistion function. 

    points : class
        Class which computes the sigma points and weights for a UKF
        algorithm. You can vary the UKF implementation by changing this
        class. For example, MerweScaledSigmaPoints implements the alpha,
        beta, kappa parameterization of Van der Merwe, and
        JulierSigmaPoints implements Julier's original kappa
        parameterization. See either of those for the required
        signature of this class if you want to implement your own.

    Attributes
    ----------

    x : numpy.array(dim_x)
        state estimate vector

    P : numpy.array(dim_x, dim_x)
        covariance estimate matrix

    z : ndarray
        Last measurement used in update(). Read only.

    R : numpy.array(dim_z, dim_z)
        measurement noise matrix

    Q : numpy.array(dim_x, dim_x)
        process noise matrix

    K : numpy.array
        Kalman gain

    y : numpy.array
        innovation residual

    inv : function, default numpy.linalg.inv
        If you prefer another inverse function, such as the Moore-Penrose
        pseudo inverse, set it to that instead:

        .. code-block:: Python

            kf.inv = np.linalg.pinv


    Examples
    --------

    Simple example of a linear order 1 kinematic filter in 2D. There is no
    need to use a UKF for this example, but it is easy to read.

    >>> def fx(x):
    >>>     # state transition function - predict next state based
    >>>     # on constant velocity model x = vt + x_0
    >>>     F = np.array([[1, 1, 0, 0],
    >>>                   [0, 1, 0, 0],
    >>>                   [0, 0, 1, 1],
    >>>                   [0, 0, 0, 1]], dtype=float)
    >>>     return np.dot(F, x)
    >>>
    >>> def hx(x):
    >>>    # measurement function - convert state into a measurement
    >>>    # where measurements are [x_pos, y_pos]
    >>>    return np.array([x[0], x[2]])
    >>>
    >>> # create sigma points to use in the filter. This is standard for Gaussian processes
    >>> points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)
    >>>
    >>> kf = UnscentedKalmanFilter(dim_x=4, dim_z=2, fx=fx, hx=hx, points=points)
    >>> kf.x = np.array([-1., 1., -1., 1]) # initial state
    >>> kf.P *= 0.2 # initial uncertainty
    >>> z_std = 0.1
    >>> kf.R = np.diag([z_std**2, z_std**2]) # 1 standard
    >>> kf.Q = Q_discrete_white_noise(dim=2, var=0.01**2, block_size=2)
    >>>
    >>> zs = [[i+randn()*z_std, i+randn()*z_std] for i in range(50)] # measurements
    >>> for z in zs:
    >>>     kf.predict()
    >>>     kf.update(z)

    For in depth explanations see my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

    Also see the filterpy/kalman/tests subdirectory for test code that
    may be illuminating.

    References
    ----------

    .. [1] Julier, Simon J. "The scaled unscented transformation,"
        American Control Converence, 2002, pp 4555-4559, vol 6.

        Online copy:
        https://www.cs.unc.edu/~welch/kalman/media/pdf/ACC02-IEEE1357.PDF

    .. [2] E. A. Wan and R. Van der Merwe, “The unscented Kalman filter for
        nonlinear estimation,” in Proc. Symp. Adaptive Syst. Signal
        Process., Commun. Contr., Lake Louise, AB, Canada, Oct. 2000.

        Online Copy:
        https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf

    .. [3] S. Julier, J. Uhlmann, and H. Durrant-Whyte. "A new method for
           the nonlinear transformation of means and covariances in filters
           and estimators," IEEE Transactions on Automatic Control, 45(3),
           pp. 477-482 (March 2000).

    .. [4] E. A. Wan and R. Van der Merwe, “The Unscented Kalman filter for
           Nonlinear Estimation,” in Proc. Symp. Adaptive Syst. Signal
           Process., Commun. Contr., Lake Louise, AB, Canada, Oct. 2000.

           https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf

    .. [5] Wan, Merle "The Unscented Kalman Filter," chapter in *Kalman
           Filtering and Neural Networks*, John Wiley & Sons, Inc., 2001.

    .. [6] R. Van der Merwe "Sigma-Point Kalman Filters for Probabilitic
           Inference in Dynamic State-Space Models" (Doctoral dissertation)
    """

    def __init__(self, dim_x, dim_z, hx, fx, points, instant_warning = False):
        """
        Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        """

        self.x = empty(dim_x)
        self.P = eye(dim_x)
        self.Q = eye(dim_x)
        self.R = eye(dim_z)
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.points_fn = points
        self.hx = hx
        self.fx = fx

        self.instant_warning    = instant_warning
        self.flag   = False

    def predict(self, fx=None, **fx_args):
        r"""
        Performs the predict step of the UKF. On return, self.x and
        self.P contain the predicted state (x) and covariance (P). '

        Important: this MUST be called before update() is called for the first
        time.

        Parameters
        ----------

        fx : callable f(x, **fx_args), optional
            State transition function. If not provided, the default
            function passed in during construction will be used.

        **fx_args : keyword arguments
            optional keyword arguments to be passed into f(x).
        """

        # calculate sigma points for given mean and covariance
        self.compute_process_sigmas(fx, **fx_args)

        #and pass sigmas through the unscented transform to compute prior
        self.x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q)


    def compute_process_sigmas(self, fx=None, **fx_args):
        """
        computes the values of sigmas_f. Normally a user would not call
        this, but it is useful if you need to call update more than once
        between calls to predict (to update for multiple simultaneous
        measurements), so the sigmas correctly reflect the updated state
        x, P.
        """

        if fx is None:
            fx = self.fx

        # calculate sigma points for given mean and covariance
        sigmas, self.Wc, self.Wm    = self.points_fn.sigma_points(self.x, self.P)

        if not hasattr(self, 'sigmas_f'):
            self.sigmas_f = empty((sigmas.shape[0], self._dim_x))
            self.sigmas_h = empty((sigmas.shape[0], self._dim_z))
        elif self.sigmas_f.shape[0] != sigmas.shape[0]:
            self.sigmas_f = empty((sigmas.shape[0], self._dim_x))
            self.sigmas_h = empty((sigmas.shape[0], self._dim_z))

        for i, s in enumerate(sigmas):

            x, flag  = fx(s, **fx_args)

            self.sigmas_f[i]    = x
            self.sigmas_h[i]    = self.hx(x)

            if flag:
                if self.flag and self.flag is not flag:
                    self.flag   = 3
                else:
                    self.flag   = flag


    def batch_filter(self, zs, Rs=None):
        """
        Performs the UKF filter over the list of measurement in `zs`.

        Parameters
        ----------

        zs : list-like
            list of measurements at each time step `self._dt` Missing
            measurements must be represented by 'None'.

        Rs : None, np.array or list-like, default=None
            optional list of values to use for the measurement error
            covariance R.

            If Rs is None then self.R is used for all epochs.

            If it is a list of matrices or a 3D array where
            len(Rs) == len(zs), then it is treated as a list of R values, one
            per epoch. This allows you to have varying R per epoch.

        Returns
        -------

        means: ndarray((n,dim_x,1))
            array of the state for each time step after the update. Each entry
            is an np.array. In other words `means[k,:]` is the state at step
            `k`.

        covariance: ndarray((n,dim_x,dim_x))
            array of the covariances for each time step after the update.
            In other words `covariance[k,:,:]` is the covariance at step `k`.

        Examples
        --------

        .. code-block:: Python

            # this example demonstrates tracking a measurement where the time
            # between measurement varies, as stored in dts The output is then smoothed
            # with an RTS smoother.

            zs = [t + random.randn()*4 for t in range (40)]

            (mu, cov, _, _) = ukf.batch_filter(zs)
            (xs, Ps, Ks) = ukf.rts_smoother(mu, cov)

        """

        try:
            z = zs[0]
        except TypeError:
            raise TypeError('zs must be list-like')

        if self._dim_z == 1:
            if not(isscalar(z) or (z.ndim == 1 and len(z) == 1)):
                raise TypeError('zs must be a list of scalars or 1D, 1 element arrays')
        else:
            if len(z) != self._dim_z:
                raise TypeError(
                    'each element in zs must be a 1D array of length {}'.format(self._dim_z))

        ## necessary to re-initialize?
        self.x = np.zeros(self._dim_x)
        self.P = eye(self._dim_x)

        z_n = np.size(zs, 0)

        # mean estimates from Kalman Filter
        means           = empty((z_n, self._dim_x))

        # state covariances from Kalman Filter
        covariances = empty((z_n, self._dim_x, self._dim_x))

        ll  = 0
        for i, z in enumerate(zs):
            self.predict()
            self.x, self.P, S, y    = update(z, self.R, self.P, self.x, self.Wc, self.Wm, self.sigmas_f, self.sigmas_h)
            means[i, :]             = self.x
            covariances[i, :, :]    = self.P
            ll  += logpdf(x=y, cov=S)

        if self.flag:
            warn('Error in transition function during filtering. Code '+str(self.flag))

        return (means, covariances, ll)

    def get_ll(self, zs, Xs, Ps):
        """
        returns likelihood based on results from smoother
        """

        ll  = 0
        for z, x, P in zip(zs, Xs, Ps):
            self.x  = x
            self.P  = P
            # self.compute_process_sigmas()
            self.predict()
            ## genauer zwischen z & x differenzieren
            ## möglicherweise braucht es doch ein predict um ein self.x zu produzieren?
            ## reihenfolge beachten. wenn z_t auf basis von z_t vorhergesagt wird ist komisch
            zp, S   = UT(self.sigmas_h, self.Wm, self.Wc, self.R)
            y       = z - zp   # residual
            ll      += logpdf(x=y, cov=S)

        if self.flag:
            warn('Error in transition function during filtering. Code '+str(self.flag))

        return ll

    def rts_smoother(self, Xs, Ps, Qs=None):
        """
        Runs the Rauch-Tung-Striebal Kalman smoother on a set of
        means and covariances computed by the UKF. The usual input
        would come from the output of `batch_filter()`.

        Parameters
        ----------

        Xs : numpy.array
           array of the means (state variable x) of the output of a Kalman
           filter.

        Ps : numpy.array
            array of the covariances of the output of a kalman filter.

        Qs: list-like collection of numpy.array, optional
            Process noise of the Kalman filter at each time step. Optional,
            if not provided the filter's self.Q will be used

        Returns
        -------

        x : numpy.ndarray
           smoothed means

        P : numpy.ndarray
           smoothed state covariances

        K : numpy.ndarray
            smoother gain at each step

        Examples
        --------

        .. code-block:: Python

            zs = [t + random.randn()*4 for t in range (40)]

            (mu, cov, _, _) = kalman.batch_filter(zs)
            (x, P, K) = rts_smoother(mu, cov, fk.F, fk.Q)
        """

        if len(Xs) != len(Ps):
            raise ValueError('Xs and Ps must have the same length')

        n, dim_x = Xs.shape

        if Qs is None:
            Qs = [self.Q] * n

        # smoother gain
        Ks = empty((n, dim_x, dim_x))

        xs, ps = Xs.copy(), Ps.copy()

        ll  = 0
        for k in reversed(range(n-1)):
            ## create sigma points from state estimate, pass through state func
            sigmas, Wc, Wm  = self.points_fn.sigma_points(xs[k], ps[k])
            num_sigmas      = sigmas.shape[0]
            sigmas_f        = empty((num_sigmas, dim_x))

            for i in range(num_sigmas):
                sigmas_f[i], flag   = self.fx(sigmas[i])
                if flag:
                    if self.instant_warning:
                        warn('Errors in transition function during smoothing. Code '+str(flag))
                    self.flag   = flag
                    if self.flag is not self.flag:
                        self.flag   = 3

            xb, Pb = UT(sigmas_f, Wm, Wc, self.Q)

            Pxb     = cross_variance(Wc, Xs[k], xb, sigmas, sigmas_f)

            # compute gain
            K = Pxb @ np.linalg.pinv(Pb)

            # update the smoothed estimates
            y   = xs[k+1] - xb
            xs[k] += K @ y
            # xs[k] += dot(K, xs[k+1] - xb)
            ps[k] += dot(K, ps[k+1] - Pb).dot(K.T)
            Ks[k] = K

            ll  += logpdf(x=y, cov=ps[k+1])
        
        if self.flag:
            warn('Errors in transition function during smoothing. Code '+str(self.flag))

        return (xs, ps, Ks, ll)

    def __repr__(self):
        return '\n'.join([
            'UnscentedKalmanFilter object',
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('Q', self.Q),
            pretty_str('R', self.R),
            pretty_str('sigmas_f', self.sigmas_f),
            pretty_str('h', self.sigmas_h),
            pretty_str('Wm', self.Wm),
            pretty_str('Wc', self.Wc),
            pretty_str('msqrt', self.msqrt),
            pretty_str('hx', self.hx),
            pretty_str('fx', self.fx),
            ])
