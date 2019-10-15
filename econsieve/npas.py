#!/bin/python
# -*- coding: utf-8 -*-

import time
import numpy as np
import pygmo as pg
from tqdm import tqdm
from grgrlib.stuff import GPP, timeprint
from .stats import logpdf


def npas(self, X=None, means=None, covs=None, get_eps=None, ngen=100, npop=10, bound_sigma=4, maxeval=0, ftol=None, method_loc=None, method_glob=None, seed=0, verbose=True):
    """Nonlinear Path-Adjustment Smoother. 
    Assumes that either, X (a time series of ensembles) is given (or can be taken from the `self` filter object), or the time series means and covs are give. From the filter object, also `eps_cov` (the diagonal matrix of the standard deviations of the shocks) and the transition function `t_func(state, shock_innovations)` must be provided.
    ...

    Parameters
    ----------
    X : array, optional
        a time series of ensembles. Either this, or means and covs have to be provided
    means : array, optional
        the series of ensemble means. Either this together with the covs, or X has to be provided
    covs : array, optional
        the series of ensemble covariances. Either this together with the means, or X has to be provided
    get_eps : function, optional
        function that, given two states (x, xp), returns a candidate solution of exogenous innovations
    ngen : int, optional
        the number of generations for each population of particle swarms. Defaults to 100
    npop : int, optional
        the number of particle swarm populations. Defaults to 10
    bound_sigma : int, optional
        the number of standard deviations included in the box constraint of the global optimizer
    maxeval : int, optional
        maximum number of function evaluations of the local nonlinear optimizer. 0 disables local optimization (default)
    ftol : float, optional
        relative maximum tolerance of the local nonlinear optimizer. Default is the default of the method.
    method_glob : function, optional
        pagmo heuristic global optimizer. Defaults to pagmo.pso
    method_loc : str, optional
        pagmo nlopt method. Defaults to 'cobyla'

    Returns
    -------
    means : array 
        the smoothed means
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

    if means is None:
        means = np.mean(X, axis=0)

    if covs is None:
        covs = np.empty((X.shape[1], X.shape[2], X.shape[2]))
        for i in range(X.shape[1]):
            covs[i, :, :] = np.cov(X[:, i, :].T)

    x = means[0]

    if method_loc is None:
        method_loc = 'cobyla'

    if method_glob is None:
        method_glob = pg.pso

    bound = np.diag(self.eps_cov)*bound_sigma

    def target(eps, x, mean, cov):

        state, flag = self.t_func(x, eps)
        if flag:
            return -np.inf

        return logpdf(state, mean=mean, cov=cov)

    if ngen:
        algo_glob = pg.algorithm(method_glob(gen=ngen, seed=seed))

    if maxeval != 0:
        algo_loc = pg.algorithm(pg.nlopt(method_loc))

        if maxeval is not None:
            algo_loc.extract(pg.nlopt).maxeval = maxeval
        if ftol is not None:
            algo_loc.extract(pg.nlopt).ftol_rel = ftol

    wrap = tqdm if verbose else lambda x: x

    ## preallocate
    res = np.empty((len(self.Z)-1,self._dim_z))
    flag = False

    for t in wrap(range(means.shape[0] - 1)):

        func = lambda eps: target(eps, x, means[t+1], covs[t+1])
        prob = pg.problem(GPP(func=func, bounds=(-bound, bound)))
        pop = pg.population(prob, npop-1-(get_eps is not None), seed=seed)

        if get_eps is not None:
            pop.push_back(get_eps(x, means[t+1]))

        pop.push_back(np.zeros(self._dim_z))

        if ngen:
            pop = algo_glob.evolve(pop)
        if maxeval != 0:
            pop = algo_loc.evolve(pop)

        eps = pop.champion_x
        x, fflag = self.t_func(x, noise=eps)

        flag |= fflag

        res[t] = eps
        means[t+1] = x

    if flag and verbose:
        print('[npas:]'.ljust(15, ' ')+'Transition function returned error.')

    if verbose:
        print('[npas:]'.ljust(15, ' ')+'Extraction took ',
              timeprint(time.time() - st, 3))

    return means, covs, res, flag
