# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-arguments

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

import numpy as np
from numba import njit

@njit(cache=True)
def unscented_transform(sigmas, Wm, Wc, noise_cov=0):
    r"""
    Computes unscented transform of a set of sigma points and weights.
    returns the mean and covariance in a tuple.

    This works in conjunction with the UnscentedKalmanFilter class.


    Parameters
    ----------

    sigmas: ndarray, of size (n, 2n+1)
        2D array of sigma points.

    Wm : ndarray [# sigmas per dimension]
        Weights for the mean.


    Wc : ndarray [# sigmas per dimension]
        Weights for the covariance.

    noise_cov : ndarray, optional
        noise matrix added to the final computed covariance matrix.


    Returns
    -------

    x : ndarray [dimension]
        Mean of the sigma points after passing through the transform.

    P : ndarray
        covariance of the sigma points after passing throgh the transform.

    Examples
    --------

    See my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    x = Wm @ sigmas

    # new covariance is the sum of the outer product of the residuals
    # times the weights
    y = sigmas - x.reshape(1,-1)
    # for i in range(y.shape[0]):
        # print(y[i])
    P = y.T*Wc @ y

    P += noise_cov

    return (x, P)
