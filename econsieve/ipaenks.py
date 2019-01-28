# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
import time, warnings

from numba import njit

from .stats import logpdf

class EnKF(object):

    def __init__(self, N, dim_x = None, dim_z = None, fx = None, hx = None, model_obj = None):

        ## get stuff directly from the model class if it exists
        if model_obj is not None:
            self._dim_x = len(model_obj.vv)
            self._dim_z = model_obj.ny
            self.fx         = model_obj.t_func
            self.hx         = model_obj.o_func

        else:
            self._dim_x = dim_x
            self._dim_z = dim_z
            self.fx     = fx
            self.hx     = hx

        self.N      = N

        self.R      = np.eye(self._dim_z)
        self.Q      = np.eye(self._dim_x)
        self.P      = np.eye(self._dim_x)

        self.x      = np.zeros(self._dim_x)

    def batch_filter(self, Z, store = False, calc_ll = False, verbose = False):

        ## store time series for later
        self.Z  = Z

        _dim_x, _dim_z, N, P, R, Q, x =     self._dim_x, self._dim_z, self.N, self.P, self.R, self.Q, self.x 

        I1  = np.ones(N)
        I2  = np.eye(N) - np.outer(I1, I1)/N

        if store:
            self.Xs              = np.empty((Z.shape[0], _dim_x, N))
            self.X_priors        = np.empty_like(self.Xs)
            self.X_bars          = np.empty_like(self.Xs)
            self.X_bar_priors    = np.empty_like(self.Xs)

        ll  = 0

        means           = np.empty((Z.shape[0], _dim_x))
        covs            = np.empty((Z.shape[0], _dim_x, _dim_x))

        Y           = np.empty((_dim_z, N))
        X_prior     = np.empty((_dim_x, N))

        mus     = np.random.multivariate_normal(mean = np.zeros(self._dim_z), cov = self.R, size=(len(Z),self.N))
        epss    = np.random.multivariate_normal(mean = np.zeros(self._dim_x), cov = self.Q, size=(len(Z),self.N))
        X       = np.random.multivariate_normal(mean = x, cov = P, size=N).T

        for nz, z in enumerate(Z):

            ## predict
            for i in range(X.shape[1]):
                eps             = epss[nz,i]
                X_prior[:,i]    = self.fx(X[:,i]+eps)[0]

            for i in range(X_prior.shape[1]):
                Y[:,i]      = self.hx(X_prior[:,i]) 

            ## update
            X_bar   = X_prior @ I2
            Y_bar   = Y @ I2
            ZZ      = np.outer(z, I1) 
            S       = np.cov(Y) + R
            X       = X_prior + X_bar @ Y_bar.T @ nl.inv((N-1)*S) @ ( ZZ - Y - mus[nz].T)

            ## storage
            means[nz,:]   = np.mean(X, axis=1)
            covs[nz,:,:]  = np.cov(X)

            if store:
                self.X_bar_priors[nz,:,:]    = X_bar
                self.X_bars[nz,:,:]          = X @ I2
                self.X_priors[nz,:,:]        = X_prior
                self.Xs[nz,:,:]              = X

            if calc_ll:
                z_mean  = np.mean(Y, axis=1)
                y   = z - z_mean
                ll      += logpdf(x = y, mean = np.zeros(_dim_z), cov = S)

        self.ll     = ll

        return means, covs, ll


    def rts_smoother(self, means = None, covs = None):

        SE      = self.Xs[-1]

        for i in reversed(range(means.shape[0] - 1)):

            J   = self.X_bars[i] @ self.X_bar_priors[i+1].T @ nl.pinv( self.X_bar_priors[i+1] @ self.X_bar_priors[i+1].T )
            SE  = self.Xs[i] + J @ (SE - self.X_priors[i+1])

            means[i]    = np.mean(SE, axis=1)
            covs[i]     = np.cov(SE)

        return means, covs


    def ipas(self, means = None, covs = None, method = None, converged_only = False, show_warnings = True, itype = (0,1), addcovs = None, return_flag = False, verbose = False):

        from scipy.optimize import minimize as so_minimize

        if method is None:
            method     = 'Powell'
        elif isinstance(method, int):
            methodl     = ["L-BFGS-B", "Nelder-Mead", "Powell", "CG", "BFGS", "TNC", "COBYLA"]
            method  = methodl[method]
            if verbose:
                print('[ipas:]'.ljust(15, ' ')+'Using %s for optimization. Available methods are %s.' %(method, ', '.join(methodl)))

        x       = means[0]
        
        itype   = np.array(itype)

        EPS     = []

        flag    = False
        flags   = False

        if verbose:
            st  = time.time()

        if 0 in itype:
            def maintarget(eps, x, mean, cov):

                state, flag     = self.fx(x, eps)

                if flag and converged_only:
                    return np.inf

                return -logpdf(state, mean = mean, cov = cov)

        if 1 in itype:
            if addcovs is None:

                if not hasattr(self, 'eps_cov'):
                    raise TypeError("Pre-smoothing requires to provide measurement and transition noise covariances ('addcovs' argument).")
            else:
                self.R          = addcovs[0]
                self.eps_cov    = addcovs[1]

            def pretarget(eps, x, obs):

                state, flag     = self.fx(x, eps)

                if flag:
                    return np.inf

                simobs  = self.hx(state)

                llobs   = -logpdf(simobs, mean = obs, cov = self.R)
                lleps   = -logpdf(eps, mean = np.zeros_like(obs), cov = self.eps_cov)

                return llobs + lleps

        superfflag  = False

        for t in range(means[:-1].shape[0]):

            eps     = np.zeros(self._dim_z)

            if 1 in itype:
                res     = so_minimize(pretarget, eps, method = method, args = (x, self.Z[t+1]))
                eps     = res['x']

            if 0 in itype:
                res     = so_minimize(maintarget, eps, method = method, args = (x, means[t+1], covs[t+1]))
                eps     = res['x']

            if not res['success']:
                if flag:
                    flags   = True
                flag    = True

            x, fflag    = self.fx(x, noise=eps)
            if fflag:
                superfflag  = True

            EPS.append(eps)

            means[t+1]  = x

        warn0 = warn1 = ''
        if superfflag:
            warn0   = 'Transitionfunction returned error.'
        if flags:
            warn1 = 'Issues(!) with convergence.'
        elif flag:
            warn1 = 'Issue with convergence'

        finflag     = False
        if flag or superfflag:
            finflag     = bool(flag) + bool(flags) + 4*bool(superfflag)
            if show_warnings:
                warnings.warn('[ipas:]'.ljust(15, ' ')+warn0+' '+warn1)

        if verbose:
            print('[ipas:]'.ljust(15, ' ')+'Extraction took ', time.time() - st, 'seconds.')

        res     = np.array(EPS)

        if return_flag:

            return means, covs, res, finflag

        return means, covs, res
