#!/bin/python
# -*- coding: utf-8 -*-

import time
import numpy as np
import pygmo as pg
from grgrlib.stuff import GPP
from .stats import logpdf

def ipas(self, X=None, means=None, covs=None, no_gen=100, maxeval=None, no_pop=10, method_loc=False, method_glob=None, bound_sigma=4, verbose=True):
    """Iterative Path-Adjusing Smoother. Assumes that either, X (a time series of ensembles) is given (or can be taken from the `self` filter object), or the time series means and covs are give. From the filter object, also `eps_cov` (the diagonal matrix of the standard deviations of the shocks) and the transition function `fx(state, shock_innovations)` must be provided.
    """

    if X is None and hasattr(self, 'Ss'):
        X = np.rollaxis(self.Ss, 2)

    if means is None:
        means = np.mean(X, axis=0)

    if covs is None:
        covs = np.empty((X.shape[1], X.shape[2], X.shape[2]))
        for i in range(X.shape[1]):
            covs[i,:,:] = np.cov(X[:,i,:].T)

    if method_loc is None:
        method_loc = 'cobyla'

    if method_glob is None:
        method_glob = pg.pso

    bound = np.diag(self.eps_cov)*bound_sigma

    EPS = []

    if verbose:
        st = time.time()

    def maintarget(eps, x, mean, cov):

        state, flag = self.fx(x, eps)

        if flag:
            return np.inf

        return logpdf(state, mean=mean, cov=cov)

    x = means[0]
    flag = False

    if method_glob:
        algo_glob = pg.algorithm(method_glob(gen = no_gen))

    if method_loc:
        algo_loc = pg.algorithm(pg.nlopt(method_loc))

        if maxeval is not None:
            algo_loc.extract(pg.nlopt).maxeval = maxeval

    for t in range(means.shape[0] - 1):

        func = lambda eps: maintarget(eps, x, means[t+1], covs[t+1])

        prob = pg.problem(GPP(func=func, bounds=(-bound, bound)))

        pop = pg.population(prob, no_pop)
        if method_glob:
            pop = algo_glob.evolve(pop)
        if method_loc:
            pop = algo_loc.evolve(pop)
        eps = pop.champion_x

        x, fflag = self.fx(x, noise=eps)

        if fflag:
            ## should never happen
            flag = True

        EPS.append(eps)
        means[t+1] = x

    if flag and verbose:
        print('[ipas:]'.ljust(15, ' ')+'Transition function returned error.')

    if verbose:
        print('[ipas:]'.ljust(15, ' ')+'Extraction took ', np.round(time.time() - st,5), 'seconds.')

    res = np.array(EPS)

    return means, covs, res, flag
