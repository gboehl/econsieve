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

# from __future__ import (absolute_import, division)

from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import eye, zeros, dot, isscalar, outer, empty
from .stats import logpdf
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
def update(z, P, x, Wc, Wm, sigmas_f, sigmas_h):
    """
    Update the UKF with the given measurements. On return,
    x and P contain the new mean and covariance of the filter.
    """

    # mean and covariance of prediction passed through unscented transform
    zp, S = unscented_transform(sigmas_h, Wm, Wc)

    # compute cross variance of the state and the measurements
    Pxz = cross_variance(Wc, x, zp, sigmas_f, sigmas_h)
    # Pxz     = Pxz[-5:]

    y = z - zp   # residual

    SI = np.linalg.pinv(S)

    K = Pxz @ SI      # Kalman gain

    # update Gaussian state estimate (x, P)
    x += K @ y
    P -= K @ S @ K.T

    return x, P


@njit(cache=True)
def unscented_transform(sigmas, Wm, Wc, noise_cov=0):
    r"""
    Computes unscented transform of a set of sigma points and weights.
    """

    x = Wm @ sigmas

    # new covariance is the sum of the outer product of the residuals times the weights
    y = sigmas - x.reshape(1, -1)
    P = y.T*Wc @ y

    P += noise_cov

    return (x, P)


class UnscentedKalmanFilter(object):

    def __init__(self, dim_x, dim_z, hx, fx, points, instant_warning=False):
        """
        Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        """

        self.x = empty(dim_x)
        self.P = eye(dim_x)
        self.Q = eye(dim_x)
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.points_fn = points
        self.hx = hx
        self.fx = fx

        self.instant_warning = instant_warning

        self._dim_sig = 0
        self.flag = False

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

        # and pass sigmas through the unscented transform to compute prior
        self.x, self.P = unscented_transform(
            self.sigmas_f, self.Wm, self.Wc, self.Q)

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

        sigmas, dim_sig = self.points_fn.sigma_points(self.x, self.P)

        if dim_sig is not self._dim_sig:
            self.Wc, self.Wm = self.points_fn.compute_weights(dim_sig)
            self._dim_sig = dim_sig

        if not hasattr(self, 'sigmas_f'):
            self.sigmas_f = empty((sigmas.shape[0], self._dim_x))
            self.sigmas_h = empty((sigmas.shape[0], self._dim_z))
        elif self.sigmas_f.shape[0] != sigmas.shape[0]:
            self.sigmas_f = empty((sigmas.shape[0], self._dim_x))
            self.sigmas_h = empty((sigmas.shape[0], self._dim_z))

        for i, s in enumerate(sigmas):

            x, flag = fx(s, **fx_args)

            self.sigmas_f[i] = x
            self.sigmas_h[i] = self.hx(x)

            if flag:
                if self.flag and self.flag is not flag:
                    self.flag = 3
                else:
                    self.flag = flag

    def batch_filter(self, zs, Rs=None):
        """
        Performs the UKF filter over the list of measurement in `zs`.
        """

        try:
            z = zs[0]
        except TypeError:
            raise TypeError('zs must be list-like')

        if self._dim_z == 1:
            if not(isscalar(z) or (z.ndim == 1 and len(z) == 1)):
                raise TypeError(
                    'zs must be a list of scalars or 1D, 1 element arrays')
        else:
            if len(z) != self._dim_z:
                raise TypeError(
                    'each element in zs must be a 1D array of length {}'.format(self._dim_z))

        # necessary to re-initialize?
        self.x = np.zeros(self._dim_x)
        # self.P = eye(self._dim_x)

        z_n = np.size(zs, 0)

        # mean estimates from Kalman Filter
        means = empty((z_n, self._dim_x))

        # state covariances from Kalman Filter
        covariances = empty((z_n, self._dim_x, self._dim_x))

        ll = 0
        for i, z in enumerate(zs):
            self.predict()
            self.x, self.P, S, y = update(
                z, self.P, self.x, self.Wc, self.Wm, self.sigmas_f, self.sigmas_h)
            means[i, :] = self.x
            covariances[i, :, :] = self.P
            ll += logpdf(x=y, cov=S)

        if self.flag:
            warn('Error in transition function during filtering. Code '+str(self.flag))

        return (means, covariances, ll)

    def rts_smoother(self, Xs, Ps, Qs=None):
        """
        Runs the Rauch-Tung-Striebal Kalman smoother on a set of
        means and covariances computed by the UKF. The usual input
        would come from the output of `batch_filter()`.
        """

        if len(Xs) != len(Ps):
            raise ValueError('Xs and Ps must have the same length')

        n, dim_x = Xs.shape
        self._dim_sig = 0

        if Qs is None:
            Qs = [self.Q] * n

        # smoother gain
        Ks = empty((n, dim_x, dim_x))

        xs, ps = Xs.copy(), Ps.copy()

        for k in reversed(range(n-1)):
            # create sigma points from state estimate, pass through state func
            # sigmas, Wc, Wm  = self.points_fn.sigma_points(xs[k], ps[k])
            sigmas, dim_sig = self.points_fn.sigma_points(xs[k], ps[k])

            if dim_sig is not self._dim_sig:
                Wc, Wm = self.points_fn.compute_weights(dim_sig)
                self._dim_sig = dim_sig

            num_sigmas = sigmas.shape[0]
            sigmas_f = empty((num_sigmas, dim_x))

            for i in range(num_sigmas):
                sigmas_f[i], flag = self.fx(sigmas[i])
                if flag:
                    if self.instant_warning:
                        warn(
                            'Errors in transition function during smoothing. Code '+str(flag))
                    self.flag = flag
                    if self.flag is not self.flag:
                        self.flag = 3

            xb, Pb = unscented_transform(sigmas_f, Wm, Wc, self.Q)

            Pxb = cross_variance(Wc, Xs[k], xb, sigmas, sigmas_f)

            # compute gain
            K = Pxb @ np.linalg.pinv(Pb)

            # update the smoothed estimates
            y = xs[k+1] - xb
            xs[k] += K @ y
            ps[k] += K @ (ps[k+1] - Pb) @ K.T
            Ks[k] = K

        if self.flag:
            warn('Errors in transition function during smoothing. Code '+str(self.flag))

        return (xs, ps, Ks)
