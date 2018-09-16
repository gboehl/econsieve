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
from scipy.linalg import cholesky
from grgrlib import nearestPSD, fast0
from .common import pretty_str

def array2d(X, dtype=None, order=None):
    """Returns at least 2-d array with data from X"""
    return np.asarray(np.atleast_2d(X), dtype=dtype, order=order)

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

    sqrt_method : function(ndarray), default=scipy.linalg.cholesky
        Defines how we compute the square root of a matrix, which has
        no unique answer. Cholesky is the default choice due to its
        speed. Typically your alternative choice will be
        scipy.linalg.sqrtm. Different choices affect how the sigma points
        are arranged relative to the eigenvectors of the covariance matrix.
        Usually this will not matter to you; if so the default cholesky()
        yields maximal performance. As of van der Merwe's dissertation of
        2004 [6] this was not a well reseached area so I have no advice
        to give you.

        If your method returns a triangular matrix it must be upper
        triangular. Do not use numpy.linalg.cholesky - for historical
        reasons it returns a lower triangular matrix. The SciPy version
        does the right thing.

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


    def __init__(self, n, alpha, beta, sqrt_method=None, subtract=None):
        #pylint: disable=too-many-arguments

        self.n = n
        self.alpha = alpha
        self.beta = beta
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()


    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2*self.n + 1


    def sigma_points(self, x, P):
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

        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, np.size(x)))

        (mu, sigma) = (x, P)

        red     = ~fast0(P, 0)
        # red     = np.ones_like(red)
        sigma   = sigma[red][:,red]
        # n_dim   = len(mu)
        n_dim   = sum(red)
        mu      = array2d(mu, dtype=float)

        ## compute sqrt(sigma)
        while True:
            try:
                sigma2  = cholesky(sigma).T
                break
            except:
                sigma   = nearestPSD(sigma)

        ## Calculate scaling factor for all off-center points
        lamda   = (self.alpha * self.alpha) * 3 - n_dim
        c       = n_dim + lamda

        ## calculate the sigma points
        points                      = np.tile(mu.T, (1, 2 * n_dim + 1))
        points[red, 1:(n_dim + 1)]    += sigma2 * np.sqrt(c)
        points[red, (n_dim + 1):]     -= sigma2 * np.sqrt(c)

        self._compute_weights(n_dim)
        Wc  = self.Wc
        Wm  = self.Wm

        return points.T, Wc, Wm


    def _compute_weights(self, n=None):
        """ Computes the weights for the scaled unscented Kalman filter.

        """

        if n is None:
            n = self.n

        lambda_ = self.alpha**2 * 3 - n

        c = .5 / (n + lambda_)
        self.Wc = np.full(2*n + 1, c)
        self.Wm = np.full(2*n + 1, c)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)

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

