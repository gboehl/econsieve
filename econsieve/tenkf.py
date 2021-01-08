#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
from grgrlib.core import tinv
from numba import njit, prange
from .stats import logpdf

try:
    import chaospy
    def multivariate_dispatch(rule):

        def multivariate(mean, cov, size):
            # rule must be of 'L', 'M', 'H', 'K' or 'S'
            res = chaospy.MvNormal(mean, cov).sample(
                size=size, rule=rule or 'L')
            res = np.moveaxis(res, 0, res.ndim-1)
            np.random.shuffle(res)
            return res

        return multivariate

except ModuleNotFoundError as e:

    def multivariate_dispatch(rule):
        def multivariate(mean, cov, size):
            return np.random.multivariate_normal(mean=mean, cov=cov, size=size)
        return multivariate

    print(str(e)+". Low-discrepancy series will not be used. This might cause a loss in precision.")

#jittable column-mean function
@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def numba_mean(x):
    res = np.empty(x.shape[0])
    for r in prange(x.shape[0]):
        res[r] = (np.mean(x[r]))
    return res

def batch_filter_jittable(Z, N, dim_x, dim_z, mus, epss, X, R, o_func, t_func, sys, precalc_mat, neps, hx):
    """Batch filter.

    Runs the TEnKF on the full dataset.
    """

    #unpack t_func Parameters
    omg, lam, x_bar = sys
    pmat, qmat, pterm, qterm, bmat, bterm = precalc_mat
    dimp, dimq = omg.shape
    dimeps = neps

    Z1 = Z.shape[0]

    I1 = np.ones(N)
    I2 = np.eye(N) - np.outer(I1, I1)/N

    # pre allocate
    Xs = np.empty((Z1, dim_x, N))
    X_priors = np.empty_like(Xs)
    X_bars = np.empty_like(Xs)
    X_bar_priors = np.empty_like(Xs)

    ll = 0

    means = np.empty((Z1, dim_x))
    covs = np.empty((Z1, dim_x, dim_x))
    Y = np.empty((dim_z, N))



    for nz, z in enumerate(Z):
        # predict
        for i in prange(N):
            i_ = np.int64(i)
            eps = epss[nz, i_]
            state, shocks = X[:, i_], eps

            if shocks is None:
                shocks = np.zeros(dimeps)

            set_k = -1
            set_l = -1


            pobs, q, l, k, flag = t_func(pmat, pterm, qmat[:, :, :-dimeps], qterm[..., :-dimeps], bmat, bterm, x_bar, *hx, state[-dimq+dimeps:], shocks, set_l, set_k, True)

            if o_func is None:

                X[:, i_] = q
                Y[:, i_] = pobs

            else:
                X[:, i_] = np.hstack((pobs, q))[0]

        if o_func is not None:
            Y = o_func(X.T).T


        X_priors[nz, :, :] = X

        # update
        X_bar = X @ I2
        Y_bar = Y @ I2
        ZZ = np.outer(z, I1)
        S = np.cov(Y) + R
        X += X_bar @ Y_bar.T @ nl.inv((N-1)*S) @ (ZZ - Y - mus[nz].T)




        X_bar_priors[nz, :, :] = X_bar
        X_bars[nz, :, :] = X @ I2
        Xs[nz, :, :] = X



        # cummulate ll

        z_mean = numba_mean(Y)

        y = z - z_mean
        ll += logpdf(x=y, mean=np.zeros(dim_z), cov=S)

    # if store:
    #     to_store =


    return X_priors, X_bar_priors, X_bars, Xs, ll

batch_jitted = njit(batch_filter_jittable, nogil=True, parallel=True, fastmath=True)#nogil=True, cache=True)


class TEnKF(object):

    name = 'TEnKF'



    def __init__(self, N, dim_x=None, dim_z=None, fx=None, hx=None, rule=None, seed=None, test=False):

        self.dim_x = np.int64(dim_x)
        self.dim_z = dim_z
        self.t_func = fx
        self.o_func = hx

        self.N = N
        self.seed = seed

        self.R = np.eye(self.dim_z)
        self.Q = np.eye(self.dim_x)
        self.P = np.eye(self.dim_x)

        self.x = np.zeros(self.dim_x)
        self.multivariate = multivariate_dispatch(rule)

        self.test = test

    def batch_filter_felber(self, Z, init_states=None, seed=None, store=False, calc_ll=False, verbose=False):
        """Batch filter.

        Runs the TEnKF on the full dataset.
        """

        # store time series for later
        self.Z = Z

        dim_x = self.dim_x
        dim_z = self.dim_z
        N = self.N


        if seed is not None:
            np.random.seed(seed)
        elif self.seed is not None:
            np.random.seed(self.seed)

        #load t_func Parameters
        # sys = self.sys
        # precalc_mat = self.precalc_mat
        # dimeps = self.neps


        mus = self.multivariate(mean=np.zeros(
            self.dim_z), cov=self.R, size=(len(Z), self.N))
        epss = self.multivariate(mean=np.zeros(
            self.dim_z), cov=self.Q, size=(len(Z), self.N))
        X = init_states or self.multivariate(mean=self.x, cov=self.P, size=N).T

        if self.fast:
            batch = batch_jitted #nogil=True, cache=True)
        else:
            batch = batch_filter_jittable



        X_priors, X_bar_priors, X_bars, Xs, result = batch(Z, N, dim_x, dim_z, mus, epss, X, self.R, self.o_func, self.t_func_jit, self.sys, self.precalc_mat, self.neps, self.hx)

        if store:
            #self.Xs = to_store
             self.X_priors, self.X_bar_priors, self.X_bars, self.Xs = X_priors, X_bar_priors, X_bars, Xs
        else:
            self.Xs = Xs

        if calc_ll:
            self.ll = result

        return result




    def batch_filter_boehl(self, Z, init_states=None, seed=None, store=False, calc_ll=False, verbose=False):
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
            X += X_bar @ Y_bar.T @ nl.inv((N-1)*S) @ (ZZ - Y - mus[nz].T)

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

    def batch_filter(self, Z, init_states=None, seed=None, store=False, calc_ll=False, verbose=False):
        """Batch filter.

        Runs the TEnKF on the full dataset.
        """
        if self.test:
            return self.batch_filter_felber(Z, init_states=init_states, seed=seed, store=store, calc_ll=calc_ll, verbose=verbose)
        else:

            return self.batch_filter_boehl(Z, init_states=init_states, seed=seed, store=store, calc_ll=calc_ll, verbose=verbose)


    def rts_smoother(self, means=None, covs=None, rcond=1e-14):

        S = self.Xs[-1]
        Ss = self.Xs.copy()

        for i in reversed(range(self.Xs.shape[0] - 1)):

            J = self.X_bars[i] @ tinv(self.X_bar_priors[i+1])
            S = self.Xs[i] + J @ (S - self.X_priors[i+1])

            Ss[i, :, :] = S

        self.Ss = Ss

        return np.rollaxis(Ss, 2)
