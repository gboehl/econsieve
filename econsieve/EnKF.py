# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
import scipy.stats as ss
import time

from .stats import logpdf

class EnsembleKalmanFilter(object):

    def __init__(self, dim_x, dim_z, fx, hx, N):

        self._dim_x = dim_x
        self._dim_z = dim_z
        self.fx     = fx
        self.hx     = hx
        self.N      = N

        self.R      = np.eye(dim_z)
        self.Q      = np.eye(dim_x)
        self.P      = np.eye(dim_x)

        self.x      = np.zeros(dim_x)

    def batch_filter(self, Z, store=True, info=False):

        I1  = np.ones(self.N)
        I2  = np.eye(self.N) - np.outer(I1, I1)/self.N

        if store:
            self.Xs              = np.empty((Z.shape[0], self._dim_x, self.N))
            self.X_priors        = np.empty_like(self.Xs)
            self.X_bars          = np.empty_like(self.Xs)
            self.X_bar_priors    = np.empty_like(self.Xs)

        ll  = 0

        means           = np.empty((Z.shape[0], self._dim_x))
        covs            = np.empty((Z.shape[0], self._dim_x, self._dim_x))

        Y           = np.empty((self._dim_z, self.N))
        X_prior     = np.empty((self._dim_x, self.N))

        mu  = ss.multivariate_normal(mean = np.zeros(self.R.shape[0]), cov = self.R, allow_singular=True)
        eps = ss.multivariate_normal(mean = np.zeros(self.Q.shape[0]), cov = self.Q, allow_singular=True)

        if info:
            st  = time.time()

        X   = ss.multivariate_normal.rvs(mean = self.x, cov = self.P, size=self.N).T

        for nz, z in enumerate(Z):

            ## predict
            for i in range(X.shape[1]):
                X_prior[:,i]    = self.fx(X[:,i]) + eps.rvs()

            for i in range(X_prior.shape[1]):
                Y[:,i]    = self.hx(X_prior[:,i]) + mu.rvs()

            ## update
            X_bar   = X_prior @ I2
            Y_bar   = Y @ I2
            Z       = np.outer(z, I1) 
            C_yy    = Y_bar @ Y_bar.T
            X       = X_prior + X_bar @ Y_bar.T @ nl.inv(C_yy + (self.N-1)*self.R) @ ( Z - Y )

            ## storage
            means[nz,:]   = np.mean(X, axis=1)
            covs[nz,:,:]  = np.cov(X)

            if store:
                self.X_bar_priors[nz,:,:]    = X_bar
                self.X_bars[nz,:,:]          = X @ I2
                self.X_priors[nz,:,:]        = X_prior
                self.Xs[nz,:,:]              = X

            z_mean  = np.mean(Y, axis=1)
            y   = z - z_mean
            S   = np.cov(Y) 
            ll  += logpdf(x=y, cov=S)

        if info:
            print('Filtering took ', time.time() - st, 'seconds.')

        self.ll     = ll

        return means, covs, ll


    def rts_smoother(self, means, covs):

        SE      = self.Xs[-1]

        for i in reversed(range(means.shape[0] - 1)):

            J   = self.X_bars[i] @ nl.pinv(self.X_bar_priors[i+1])
            SE  = self.Xs[i] + J @ (SE - self.X_priors[i+1])

            means[i]    = np.mean(SE, axis=1)
            covs[i]     = np.cov(SE)

        return means, covs
