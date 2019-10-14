#!/bin/python
# -*- coding: utf-8 -*-

import time
import numpy as np
import pygmo as pg
from grgrlib.stuff import GPP, timeprint
from .stats import logpdf


def ipas(self, X=None, get_eps=None, refilter=False, means=None, covs=None, ngen=100, npop=10, maxeval=0, ftol=None, method_loc=None, method_glob=None, bound_sigma=4, seed=0, verbose=True):
    """Iterative Path-Adjusing Smoother. Assumes that either, X (a time series of ensembles) is given (or can be taken from the `self` filter object), or the time series means and covs are give. From the filter object, also `eps_cov` (the diagonal matrix of the standard deviations of the shocks) and the transition function `t_func(state, shock_innovations)` must be provided.
    """

    if verbose:
        st = time.time()

    if refilter:
        ## X must be an ensembles of x dimensions
        if X is None:
            X = self.Ss[0]

        means = np.empty((len(self.Z), X.shape[0]))
        covs = np.empty((len(self.Z), X.shape[0], X.shape[0]))
        Xs = np.empty((len(self.Z), X.shape[0], self.N))

        means[0] = np.mean(X, axis=1)
        covs[0] = np.cov(X)
        Xs[0] = X

        I1 = np.ones(self.N)
        I2 = np.eye(self.N) - np.outer(I1, I1)/self.N

        Y = np.empty((self._dim_z, self.N))
    else:
        ## X must be time series of ensembles of x dimensions
        if X is None and hasattr(self, 'Ss'):
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

    def maintarget(eps, x, mean, cov):

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

    if verbose:
        from tqdm import tqdm
        wrap = tqdm
    else:
        def wrap(x): return x

    ## preallocate
    res = np.empty((len(self.Z)-1,self._dim_z))
    flag = False

    for t in wrap(range(means.shape[0] - 1)):

        if refilter:
            # predict
            for i in range(X.shape[1]):
                eps = np.random.multivariate_normal(mean=np.zeros(self._dim_z), cov=self.Q)
                X[:, i] = self.t_func(X[:, i], eps)[0]

            Y = self.o_func(X.T).T

            mus = np.random.multivariate_normal(mean=np.zeros(self._dim_z), cov=self.R, size=self.N)

            # # update
            X_bar = X @ I2
            Y_bar = Y @ I2
            ZZ = np.outer(self.Z[t+1], I1)
            S = np.cov(Y) + self.R
            X += X_bar @ Y_bar.T @ np.linalg.inv((self.N-1)*S) @ (ZZ - Y - mus.T)

            mean = np.mean(X, axis=1)
            cov = np.cov(X)
        else:
            mean = means[t+1]
            cov = covs[t+1]

        if get_eps is not None:
            guess = get_eps(x, mean)

        def func(eps): 
            return maintarget(eps, x, mean, cov)

        prob = pg.problem(GPP(func=func, bounds=(-bound, bound)))
        pop = pg.population(prob, npop-1-(get_eps is not None), seed=seed)

        if get_eps is not None:
            pop.push_back(guess)

        pop.push_back(np.zeros(self._dim_z))

        if ngen:
            pop = algo_glob.evolve(pop)
        if maxeval != 0:
            pop = algo_loc.evolve(pop)

        eps = pop.champion_x
        x, fflag = self.t_func(x, noise=eps)

        if fflag:
            # should never happen
            flag = True

        res[t] = eps
        means[t+1] = x

        if refilter:
            X = (X.T + x - np.mean(X, axis=1)).T
            Xs[t+1] = X
            covs[t+1] = cov

    if flag and verbose:
        print('[ipas:]'.ljust(15, ' ')+'Transition function returned error.')

    if verbose:
        print('[ipas:]'.ljust(15, ' ')+'Extraction took ',
              timeprint(time.time() - st, 3))

    if refilter:
        self.EXs = Xs

    return means, covs, res, flag
