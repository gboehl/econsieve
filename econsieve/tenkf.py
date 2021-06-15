#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import sqrtm
from grgrlib.la import tinv, nearest_psd
from numba import njit
from .stats import logpdf

try:
    import chaospy

    if hasattr(chaospy.distributions.kernel.baseclass, 'Dist'):
        def init_mv_normal(self, loc=[0, 0], scale=[[1, .5], [.5, 1]]):

            loc = np.asfarray(loc)
            scale = np.asfarray(scale)
            assert len(loc) == len(scale)
            self._repr = {"loc": loc.tolist(), "scale": scale.tolist()}

            try:
                C = np.linalg.cholesky(scale)
                Ci = np.linalg.inv(C)

            except np.linalg.LinAlgError as err:
                C = np.real(sqrtm(scale))
                Ci = np.linalg.pinv(C)

            chaospy.baseclass.Dist.__init__(self, C=C, Ci=Ci, loc=loc)
     
        # must be patched to allow for a covariance that is only PSD
        chaospy.MvNormal.__init__ = init_mv_normal

    else:

        def init_mv_normal(
                self,
                dist,
                mean=0,
                covariance=1,
                rotation=None,
                repr_args=None,
        ):
            mean = np.atleast_1d(mean)
            length = max(len(dist), len(mean), len(covariance))

            exclusion = dist._exclusion.copy()
            dist = chaospy.Iid(dist, length)

            covariance = np.asarray(covariance)

            rotation = [key for key, _ in sorted(enumerate(dist._dependencies), key=lambda x: len(x[1]))]

            accumulant = set()
            dependencies = [deps.copy() for deps in dist._dependencies]
            for idx in rotation:
                accumulant.update(dist._dependencies[idx])
                dependencies[idx] = accumulant.copy()

            self._permute = np.eye(len(rotation), dtype=int)[rotation]
            self._covariance = covariance
            self._pcovariance = self._permute.dot(covariance).dot(self._permute.T)
            try:
                cholesky = np.linalg.cholesky(self._pcovariance)
                self._fwd_transform = self._permute.T.dot(np.linalg.inv(cholesky))

            except np.linalg.LinAlgError as err:
                cholesky = np.real(sqrtm(self._pcovariance))
                self._fwd_transform = self._permute.T.dot(np.linalg.pinv(cholesky))

            self._inv_transform = self._permute.T.dot(cholesky)
            self._dist = dist

            super(chaospy.distributions.MeanCovarianceDistribution, self).__init__(
                parameters=dict(mean=mean, covariance=covariance),
                dependencies=dependencies,
                rotation=rotation,
                exclusion=exclusion,
                repr_args=repr_args,
            )

        def get_parameters_patched(self, idx, cache, assert_numerical=True):
            # avoids all functionality not used

            parameters = super(chaospy.distributions.MeanCovarianceDistribution, self).get_parameters(
                idx, cache, assert_numerical=assert_numerical)

            mean = parameters["mean"]

            mean = mean[self._rotation]

            dim = self._rotation.index(idx)

            return dict(idx=idx, mean=mean, sigma=None, dim=dim, mut=None, cache=cache)

        # must be patched to allow for a covariance that is only PSD
        chaospy.distributions.MeanCovarianceDistribution.__init__ = init_mv_normal
        chaospy.distributions.MeanCovarianceDistribution.get_parameters = get_parameters_patched


    def multivariate_dispatch(rule):

        def multivariate(mean, cov, size):
            # rule must be of 'L', 'M', 'H', 'K' or 'S'

            res = chaospy.MvNormal(mean, cov).sample(size=size, rule=rule or 'L')
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


class TEnKF(object):

    name = 'TEnKF'

    def __init__(self, N, dim_x=None, dim_z=None, fx=None, hx=None, rule=None, seed=None):

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
        self.multivariate = multivariate_dispatch(rule)

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
            X += X_bar @ Y_bar.T @ np.linalg.inv((N-1)*S) @ (ZZ - Y - mus[nz].T)

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
