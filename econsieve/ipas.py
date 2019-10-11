#!/bin/python
# -*- coding: utf-8 -*-

import time
import numpy as np
import pygmo as pg
from grgrlib.stuff import GPP, timeprint
from .stats import logpdf


def ipas(self, X=None, get_eps=None, means=None, covs=None, ngen=100, npop=10, maxeval=0, ftol=None, method_loc=None, method_glob=None, bound_sigma=4, seed=0, verbose=True):
    """Iterative Path-Adjusing Smoother. Assumes that either, X (a time series of ensembles) is given (or can be taken from the `self` filter object), or the time series means and covs are give. From the filter object, also `eps_cov` (the diagonal matrix of the standard deviations of the shocks) and the transition function `fx(state, shock_innovations)` must be provided.
    """

    if verbose:
        st = time.time()

    if X is None and hasattr(self, 'Ss'):
        X = np.rollaxis(self.Ss, 2)

    if means is None:
        means = np.mean(X, axis=0)

    res = np.empty((means.shape[0]-1,self._dim_z))

    ## presmoothing
    if get_eps is not None:
        for i in range(res.shape[0]):
            res[i,:] = get_eps(means[i], means[i+1])

    if covs is None:
        covs = np.empty((X.shape[1], X.shape[2], X.shape[2]))
        for i in range(X.shape[1]):
            covs[i, :, :] = np.cov(X[:, i, :].T)

    if method_loc is None:
        method_loc = 'cobyla'

    if method_glob is None:
        method_glob = pg.pso

    bound = np.diag(self.eps_cov)*bound_sigma

    def maintarget(eps, x, mean, cov):

        state, flag = self.fx(x, eps)

        if flag:
            return -np.inf

        return logpdf(state, mean=mean, cov=cov)

    x = means[0]
    flag = False

    if ngen:
        algo_glob = pg.algorithm(method_glob(gen=ngen, seed=seed))

    if maxeval != 0:
        algo_loc = pg.algorithm(pg.nlopt(method_loc))

        if maxeval is not None:
            algo_loc.extract(pg.nlopt).maxeval = maxeval
        if ftol is not None:
            algo_loc.extract(pg.nlopt).ftol_rel = ftol

    if verbose:
        from tqdm import tqdm
        wrap = tqdm
    else:
        def wrap(x): return x

    for t in wrap(range(means.shape[0] - 1)):

        def func(eps): 
            return maintarget(eps, x, means[t+1], covs[t+1])

        prob = pg.problem(GPP(func=func, bounds=(-bound, bound)))

        pop = pg.population(prob, npop-1-(get_eps is not None), seed=seed)

        if get_eps is not None:
            pop.push_back(res[t])

        pop.push_back(np.zeros(self._dim_z))

        if ngen:
            pop = algo_glob.evolve(pop)
        if maxeval != 0:
            pop = algo_loc.evolve(pop)

        eps = pop.champion_x
        x, fflag = self.fx(x, noise=eps)

        if fflag:
            # should never happen
            flag = True

        res[t,:] = eps
        means[t+1] = x

    if flag and verbose:
        print('[ipas:]'.ljust(15, ' ')+'Transition function returned error.')

    if verbose:
        print('[ipas:]'.ljust(15, ' ')+'Extraction took ',
              timeprint(time.time() - st, 3))

    return means, covs, res, flag
