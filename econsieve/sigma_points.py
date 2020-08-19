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

from __future__ import division
import numpy as np
from numba import njit
from grgrlib import cholesky


@njit(cache=True)
def _compute_weights(n, alpha, beta):
    """ Computes the weights for the scaled unscented Kalman filter.

    """

    lamb = alpha**2 * 3 - n

    c = .5 / (n + lamb)
    Wc = np.full(2*n + 1, c)
    Wm = np.full(2*n + 1, c)
    Wc[0] = lamb / (n + lamb) + (1 - alpha**2 + beta)
    Wm[0] = lamb / (n + lamb)

    return Wc, Wm


@njit(cache=True)
def sigma_points(x, P, alpha, beta):

    # jittified version of fast0 from grgrlib
    P_ic = np.abs(P) > 10e-5
    red = np.sum(P_ic, axis=0) > 0
    red = np.ones_like(red)

    P2 = P[red][:, red]
    n = np.sum(red)

    # compute sqrt(P)
    P3 = cholesky(P2) * alpha * np.sqrt(3)

    points = np.empty((2*n + 1, len(x)))

    points[:] = x

    for k in range(n):
        points[k+1][red] += P3[k]
        points[n+k+1][red] -= P3[k]

    return points, n


class ScaledSigmaPoints(object):

    def __init__(self, n, alpha, beta, subtract=None):

        self.compute_weights = lambda n: _compute_weights(n, alpha, beta)
        self.sigma_points = lambda x, P: sigma_points(x, P, alpha, beta)
