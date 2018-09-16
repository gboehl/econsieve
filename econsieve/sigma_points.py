# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-instance-attributes

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

from __future__ import division
import numpy as np
from .common import pretty_str
from numba import njit
from grgrlib import cholesky

@njit(cache=True)
def _compute_weights(n, alpha, beta):
    """ Computes the weights for the scaled unscented Kalman filter.

    """

    lambda_ = alpha**2 * 3 - n

    c = .5 / (n + lambda_)
    Wc = np.full(2*n + 1, c)
    Wm = np.full(2*n + 1, c)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    Wm[0] = lambda_ / (n + lambda_)

    return Wc, Wm

@njit(cache=True)
def sigma_points(x, P, alpha, beta):
    """ Computes the sigma points for an unscented Kalman filter
    given the mean (x) and covariance(P) of the filter.
    Returns tuple of the sigma points and weights.

    Works with both scalar and array inputs:
    sigma_points (5, 9, 2) # mean 5, covariance 9
    sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I

    Parameters
    ----------

    x : An array-like object of the means of length n
        Can be a scalar if 1D.
        examples: 1, [1,2], np.array([1,2])

    P : scalar, or np.array
       Covariance of the filter. If scalar, is treated as eye(n)*P.

    Returns
    -------

    sigmas : np.array, of size (n, 2n+1)
        Two dimensional array of sigma points. Each column contains all of
        the sigmas for one dimension in the problem space.

        Ordered by Xi_0, Xi_{1..n}, Xi_{n+1..2n}
    """

    (mu, sigma) = (x, P)

    ## jittified version of fast0 from grgrlib
    P_ic    = np.abs(P) > 10e-5
    red     = np.sum(P_ic, axis=0) > 0
    # red     = np.ones_like(red)

    sigma   = sigma[red][:,red]
    n       = np.sum(red)

    ## compute sqrt(sigma)
    sigma2  = cholesky(sigma)

    ## Calculate scaling factor for all off-center points
    lamda   = alpha**2  * 3 - n
    c       = n + lamda

    points  = np.empty((len(mu), 2*n + 1))
    for i in range(2*n + 1):
        points[:,i]     = mu

    points[red, 1:(n + 1)]    += sigma2 * np.sqrt(c)
    points[red, (n + 1):]     -= sigma2 * np.sqrt(c)

    Wc, Wm  = _compute_weights(n, alpha, beta)

    return points.T, Wc, Wm

class GreedyMerweScaledSigmaPoints(object):

    """
    Generates sigma points and weights according to Van der Merwe's
    2004 dissertation[1] for the UnscentedKalmanFilter class.. It
    parametizes the sigma points using alpha, beta, kappa terms, and
    is the version seen in most publications.

    Unless you know better, this should be your default choice.

    Parameters
    ----------

    n : int
        Dimensionality of the state. 2n+1 weights will be generated.

    alpha : float
        Determins the spread of the sigma points around the mean.
        Usually a small positive value (1e-3) according to [3].

    beta : float
        Incorporates prior knowledge of the distribution of the mean. For
        Gaussian x beta=2 is optimal, according to [3].

    kappa : float, default=0.0
        Secondary scaling parameter usually set to 0 according to [4],
        or to 3-n according to [5].

    subtract : callable (x, y), optional
        Function that computes the difference between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars.

    Attributes
    ----------

    Wm : np.array
        weight for each sigma point for the mean

    Wc : np.array
        weight for each sigma point for the covariance

    Examples
    --------

    See my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python


    References
    ----------

    .. [1] R. Van der Merwe "Sigma-Point Kalman Filters for Probabilitic
           Inference in Dynamic State-Space Models" (Doctoral dissertation)

    """


    def __init__(self, n, alpha, beta, subtract=None):
        #pylint: disable=too-many-arguments

        self.n = n
        self.alpha = alpha
        self.beta = beta

        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        # self.compute_weights   = lambda n: self._compute_weights(n, alpha, beta)
        self.compute_weights    = lambda n: _compute_weights(n, alpha, beta)
        self.sigma_points       = lambda x,P : sigma_points(x, P, alpha, beta)
        # self._compute_weights(n)


    def __repr__(self):

        return '\n'.join([
            'MerweScaledSigmaPoints object',
            pretty_str('n', self.n),
            pretty_str('alpha', self.alpha),
            pretty_str('beta', self.beta),
            pretty_str('Wm', self.Wm),
            pretty_str('Wc', self.Wc),
            pretty_str('subtract', self.subtract),
            pretty_str('sqrt', self.sqrt)
            ])

