#!/bin/python
# -*- coding: utf-8 -*-

import time
import numpy as np
from tqdm import tqdm
from grgrlib.core import timeprint
from grgrlib.optimize import cmaes
from .stats import logpdf


def npas(self, X=None, vals=None, covs=None, get_eps=None, nsamples=False, bound_sigma=4, frtol=1e-5, seed=0, verbose=True, **cmaes_args):
    """Nonlinear Path-Adjustment Smoother. 

    Assumes that either `X` (a time series of ensembles) is given (or can be taken from the filter `self` object), or that the time series vals and covs are given. From the filter object, also `eps_cov` (the diagonal matrix of the standard deviations of the shocks) and the transition function `t_func(state, shock_innovations)` must be provided.
    ...

    Parameters
    ----------
    X : array, optional
        a time series of ensembles. Either this, or vals and covs have to be provided
    vals : array, optional
        the series of ensemble values, probably you want the means. Either this together with the covs, or X has to be provided
    covs : array, optional
        the series of ensemble covariances. Either this together with `vals`, or `X` has to be provided
    get_eps : function, optional
        function that, given two states (x, xp), returns a candidate solution of exogenous innovations
    bound_sigma : int, optional
        the number of standard deviations included in the box constraint of the global optimizer

    Returns
    -------
    X : array 
        the smoothed vals
    covs : array 
        the covariances
    res : array 
        the smoothed/estimated exogenousinnovations
    flag : bool
        an error flag of the transition function
    """

    if verbose:
        st = time.time()

    ## X must be time series of ensembles of x dimensions
    if X is None:
        X = np.rollaxis(self.Ss, 2)

    if covs is None:
        covs = np.empty((X.shape[1], X.shape[2], X.shape[2]))
        for i in range(X.shape[1]):
            covs[i, :, :] = np.cov(X[:, i, :].T)

    bound = np.diag(self.eps_cov)*bound_sigma

    def target(eps, x, mean, cov):

        state, flag = self.t_func(x, 2*bound*eps)
        if flag:
            return np.inf

        return -logpdf(state, mean=mean, cov=cov)

    wrap = tqdm if verbose else lambda x: x
    owrap = wrap if nsamples else lambda x: x
    iwrap = wrap if nsamples else lambda x: x

    # the smooth version to do this would be rejection sampling. But as max(p(x) / q(x)) is unknown and expensive to evaluate, rejection sampling would likewise be expensive

    if not nsamples:
        X[0] = np.mean(X, axis=0)
        nsamples = 1
    else:
        np.random.shuffle(X)

    # preallocate
    res = np.empty((nsamples, len(self.Z)-1,self._dim_z))
    flag = False

    for n,s in enumerate(owrap(X[:nsamples])):

        x = X[n][0]

        for t in iwrap(range(s.shape[0] - 1)):

            func = lambda eps: target(eps, x, s[t+1], covs[t+1])

            eps0 = get_eps(x, s[t+1])/bound/2 if get_eps else np.zeros(self._dim_z)

            res_cma = cmaes(func, eps0, 0.1, verbosity=0, frtol=frtol, **cmaes_args)
            eps = res_cma[0]*bound*2

            x, fflag = self.t_func(x, noise=eps)

            flag |= fflag

            res[n][t] = eps
            X[n][t+1] = x

        if flag and verbose:
            print('[npas:]'.ljust(15, ' ')+'Transition function returned error.')

        if not nsamples and verbose:
            print('[npas:]'.ljust(15, ' ')+'Extraction took ',
                  timeprint(time.time() - st, 3))

    return X[:nsamples], covs, res, flag
