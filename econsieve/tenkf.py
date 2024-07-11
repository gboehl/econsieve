#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from functools import partial
from scipy.linalg import sqrtm
from scipy.stats.qmc import MultivariateNormalQMC, LatinHypercube
from grgrlib.linalg import tinv, nearest_psd
from .stats import logpdf


def multivariate_dispatch(engine):

    def multivariate(mean, cov, size):
        ndim = len(mean)
        size = (size,) if isinstance(size, int) else size
        nsample = np.prod(size)

        # define target distribution
        scov = np.real(sqrtm(cov))
        dist = MultivariateNormalQMC(mean, cov_root=scov, engine=engine(ndim))

        # draw sample
        res = dist.random(nsample)
        return np.reshape(res, (*size, ndim))

    return multivariate


class TEnKF(object):

    name = 'TEnKF'

    def __init__(self, N, dim_x=None, dim_z=None, fx=None, hx=None, qmc_engine=None, seed=None):

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.t_func = fx
        self.o_func = hx

        self.N = N
        self.seed = seed

        self.R = np.eye(self.dim_z)
        self.Q = np.eye(self.dim_x)
        self.P = np.eye(self.dim_x)
        self.x = np.zeros(self.dim_x)

        # enforce deterministic sampling by default
        seed_default_qmc_engine = 0 if seed is None else seed
        qmc_engine = partial(LatinHypercube, seed=seed_default_qmc_engine) if qmc_engine is None else qmc_engine
        self.multivariate = multivariate_dispatch(qmc_engine)


    def batch_filter(self, Z, init_states=None, seed=None, store=False, calc_ll=False, verbose=False):
        """Batch filter.

        Runs the TEnKF on the full dataset.
        """

        # store time series for later
        self.Z = Z

        dim_x = self.dim_x
        dim_z = self.dim_z
        N = self.N

        I1 = np.ones(N)
        I2 = np.eye(N) - np.outer(I1, I1)/N

        # pre allocate
        if store:
            self.Xs = np.empty((Z.shape[0], dim_x, N))
            self.X_priors = np.empty_like(self.Xs)
            self.X_bars = np.empty_like(self.Xs)
            self.X_bar_priors = np.empty_like(self.Xs)

        ll = 0

        if seed is not None:
            np.random.seed(seed)
        elif self.seed is not None:
            np.random.seed(self.seed)

        means = np.empty((Z.shape[0], dim_x))
        covs = np.empty((Z.shape[0], dim_x, dim_x))
        Y = np.empty((dim_z, N))

        mus = self.multivariate(mean=np.zeros(
            self.dim_z), cov=self.R, size=(len(Z), self.N))
        epss = self.multivariate(mean=np.zeros(
            self.dim_z), cov=self.Q, size=(len(Z), self.N))
        X = init_states or self.multivariate(mean=self.x, cov=self.P, size=N).T

        self.Xs = np.empty((Z.shape[0], dim_x, N))

        for nz, z in enumerate(Z):

            # predict
            for i in range(N):
                eps = epss[nz, i]
                if self.o_func is None:
                    X[:, i], Y[:, i] = self.t_func(X[:, i], eps)[0]
                else:
                    X[:, i] = self.t_func(X[:, i], eps)[0]

            if self.o_func is not None:
                Y = self.o_func(X.T).T

            if store:
                self.X_priors[nz, :, :] = X

            # update
            X_bar = X @ I2
            Y_bar = Y @ I2
            ZZ = np.outer(z, I1)
            S = np.cov(Y) + self.R
            X += X_bar @ Y_bar.T @ np.linalg.inv((N-1)
                                                 * S) @ (ZZ - Y - mus[nz].T)

            if store:
                self.X_bar_priors[nz, :, :] = X_bar
                self.X_bars[nz, :, :] = X @ I2
                self.Xs[nz, :, :] = X

            if calc_ll:
                # cummulate ll
                z_mean = np.mean(Y, axis=1)
                y = z - z_mean
                ll += logpdf(x=y, mean=np.zeros(dim_z), cov=S)
            else:
                self.Xs[nz, :, :] = X

        if calc_ll:
            self.ll = ll
            return ll
        else:
            return np.rollaxis(self.Xs, 2)

    def rts_smoother(self, means=None, covs=None, rcond=1e-14):

        S = self.Xs[-1]
        Ss = self.Xs.copy()

        for i in reversed(range(self.Xs.shape[0] - 1)):

            J = self.X_bars[i] @ tinv(self.X_bar_priors[i+1])
            S = self.Xs[i] + J @ (S - self.X_priors[i+1])

            Ss[i, :, :] = S

        self.Ss = Ss

        return np.rollaxis(Ss, 2)
