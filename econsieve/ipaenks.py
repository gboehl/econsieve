# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
import time
from scipy.optimize import minimize as so_minimize

from numba import njit

from .stats import logpdf


class EnKF(object):

    def __init__(self, N, dim_x=None, dim_z=None, fx=None, hx=None, model_obj=None):

        self._dim_x = dim_x
        self._dim_z = dim_z
        self.fx = fx
        self.hx = hx

        self.N = N

        self.R = np.eye(self._dim_z)
        self.Q = np.eye(self._dim_x)
        self.P = np.eye(self._dim_x)

        self.x = np.zeros(self._dim_x)

    def batch_filter(self, Z, store=False, calc_ll=False, verbose=False):

        # store time series for later
        self.Z = Z

        _dim_x, _dim_z, N, P, R, Q, x = self._dim_x, self._dim_z, self.N, self.P, self.R, self.Q, self.x

        I1 = np.ones(N)
        I2 = np.eye(N) - np.outer(I1, I1)/N

        if store:
            self.Xs = np.empty((Z.shape[0], _dim_x, N))
            self.X_priors = np.empty_like(self.Xs)
            self.X_bars = np.empty_like(self.Xs)
            self.X_bar_priors = np.empty_like(self.Xs)

        ll = 0

        means = np.empty((Z.shape[0], _dim_x))
        covs = np.empty((Z.shape[0], _dim_x, _dim_x))

        Y = np.empty((_dim_z, N))
        X_prior = np.empty((_dim_x, N))

        mus = np.random.multivariate_normal(
            mean=np.zeros(self._dim_z), cov=self.R, size=(len(Z), self.N))
        epss = np.random.multivariate_normal(
            mean=np.zeros(self._dim_x), cov=self.Q, size=(len(Z), self.N))
        X = np.random.multivariate_normal(mean=x, cov=P, size=N).T

        for nz, z in enumerate(Z):

            # predict
            for i in range(X.shape[1]):
                eps = epss[nz, i]
                X_prior[:, i] = self.fx(X[:, i]+eps)[0]

            for i in range(X_prior.shape[1]):
                Y[:, i] = self.hx(X_prior[:, i])

            # update
            X_bar = X_prior @ I2
            Y_bar = Y @ I2
            ZZ = np.outer(z, I1)
            S = np.cov(Y) + R
            X = X_prior + \
                X_bar @ Y_bar.T @ nl.inv((N-1)*S) @ (ZZ - Y - mus[nz].T)


            if store:
                self.X_bar_priors[nz, :, :] = X_bar
                self.X_bars[nz, :, :] = X @ I2
                self.X_priors[nz, :, :] = X_prior
                self.Xs[nz, :, :] = X

            if calc_ll:
                # cummulate ll
                z_mean = np.mean(Y, axis=1)
                y = z - z_mean
                ll += logpdf(x=y, mean=np.zeros(_dim_z), cov=S)
            else:
                # storage of means & cov
                means[nz, :] = np.mean(X, axis=1)
                covs[nz, :, :] = np.cov(X)


        if calc_ll:
            self.ll = ll
            return ll
        else:
            return means, covs

    def rts_smoother(self, means=None, covs=None, rcond=1e-14):

        SE = self.Xs[-1]

        for i in reversed(range(means.shape[0] - 1)):

            J = self.X_bars[i] @ self.X_bar_priors[i+1].T @ nl.pinv(
                self.X_bar_priors[i+1] @ self.X_bar_priors[i+1].T, rcond=rcond)
            SE = self.Xs[i] + J @ (SE - self.X_priors[i+1])

            means[i] = np.mean(SE, axis=1)
            covs[i] = np.cov(SE)

        return means, covs

    def ipas(self, means=None, covs=None, method=None, penalty=False, show_warnings=True, itype=None, presmoothing=None, objects=None, min_options=None, return_flag=False, verbose=False):
        """
        itype legend:
            0: MLE (pre-)smoothing
            1: IPA-smoothing
        presmooter:
            can take 'residuals' (default), 'linear' or 'off'
            if 'linear', objects must contain a touple of the objects T1, T2 & T3 s.t.
            z_t  = T1 @ x_{t-1} + T2 @ eps + T3
        objects:
            objects assume the form ((T1, T3, T3), eps_cov, x2eps). If any of these objects is not needed just pack a None or any other random element.
        """

        if method is None:

            # yields most roboust results
            method = 'Nelder-Mead'

            if min_options is None:
               min_options = {'maxfev': 30000}

            if presmoothing is None and objects is not None:
               presmoothing='linear'

        elif isinstance(method, int):

            methodl = ["Nelder-Mead", "BFGS", "Powell",
                       "L-BFGS-B", "CG", "TNC", "COBYLA"]
            method = methodl[method]

            if verbose:
                print('[ipas:]'.ljust(
                    15, ' ')+'Using %s for optimization. Available methods are %s.' % (method, ', '.join(methodl)))

        x = means[0]

        if itype is None:
            itype = 1
        itype = np.array(itype)

        EPS = []

        flag = False
        flags = False

        if verbose:
            st = time.time()

        if presmoothing is None:
            presmoothing = 'off'

        if presmoothing is not 'off' or 0 in itype:
            if objects is None:
                raise TypeError(
                    "(MLE-)Presmoothing requires to provide additional objects ('objects' argument). Form: (T1, T2, T3), eps_cov, x2eps")
            else:
                (T1, T2, T3), x2eps = objects

        if presmoothing == 'linear':
            Ce = nl.inv(self.eps_cov)
            Cz = nl.inv(self.R)

        if 0 in itype:
            def pretarget(eps, x, obs):

                state, flag = self.fx(x, eps)

                simobs = self.hx(state)

                llobs = -logpdf(simobs, mean=obs, cov=self.R)
                lleps = -logpdf(eps, mean=np.zeros_like(obs), cov=self.eps_cov)

                if flag:
                    return llobs + lleps + penalty

                return llobs + lleps

        if 1 in itype:
            def maintarget(eps, x, mean, cov):

                state, flag = self.fx(x, eps)

                if flag:
                    return -logpdf(state, mean=mean, cov=cov) + penalty

                return -logpdf(state, mean=mean, cov=cov)

        superfflag = False

        for t in range(means[:-1].shape[0]):

            if presmoothing == 'off':
                eps = np.zeros(self._dim_z)
            elif presmoothing == 'residuals':
                eps = (means[t+1] - self.fx(means[t],
                                            np.zeros(self._dim_z))[0]) @ x2eps
            elif presmoothing == 'linear':
                eps = nl.pinv(Ce + T2.T @ Cz @
                              T2) @ T2.T @ Cz @ (self.Z[t+1] - T1 @ x - T3)
            else:
                raise Exception('[ipas:]'.ljust(15, ' ')+" 'Presmoothing' must be either 'off', residuals' or 'linear'.")

            if 0 in itype:
                res = so_minimize(pretarget, eps, method=method, args=(
                    x, self.Z[t+1]), options=min_options)
                eps = res['x']

            if 1 in itype:
                res = so_minimize(maintarget, eps, method=method, args=(
                    x, means[t+1], covs[t+1]), options=min_options)
                eps = res['x']

            if (0, 1) in itype:
                if not res['success']:
                    if verbose:
                        print('[ipas -> minimize:]'.ljust(30, ' ')+res['message'])
                    if flag:
                        flags = True
                    flag = True

            x, fflag = self.fx(x, noise=eps)
            if fflag:
                superfflag = True

            EPS.append(eps)

            means[t+1] = x

        warn0 = warn1 = ''
        if superfflag:
            warn0 = 'Transition function returned error.'
        if flags:
            warn1 = "Issues with convergence of 'minimize'."
        elif flag:
            warn1 = "Issue with convergence of 'minimize'."

        finflag = False
        if flag or superfflag:
            finflag = bool(flag) + bool(flags) + 4*bool(superfflag)
            if show_warnings:
                print('[ipas:]'.ljust(15, ' ')+warn0+' '+warn1)

        if verbose:
            print('[ipas:]'.ljust(15, ' ')+'Extraction took ',
                  np.round(time.time() - st,5), 'seconds.')

        res = np.array(EPS)

        if return_flag:
            return means, covs, res, finflag

        return means, covs, res
