#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
import time
from scipy.optimize import minimize as so_minimize
from .stats import logpdf

def ipas(filter_obj=None, means=None, covs=None, method=None, penalty=False, show_warnings=True, itype=None, presmoothing=None, objects=None, min_options=None, return_flag=False, verbose=False):
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
    ## idea is that either the filter object OR the respective contents must be provided

    if method is None:

        # yields most roboust results
        method = 'Nelder-Mead'

        if min_options is None:
           min_options = {'maxfev': 30000}

        if presmoothing is None and objects is not None:
           presmoothing='residuals'

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

    if not presmoothing or presmoothing is None:
        presmoothing = 'off'

    if presmoothing is not 'off' or 0 in itype:
        if objects is None:
            raise TypeError(
                "(MLE-)Presmoothing requires to provide additional objects ('objects' argument). Form: (T1, T2, T3), eps_cov, x2eps")
        else:
            (T1, T2, T3), x2eps = objects

    if presmoothing == 'linear':
        Ce = nl.inv(filter_obj.eps_cov)
        Cz = nl.inv(filter_obj.R)

    if 0 in itype:
        def pretarget(eps, x, obs):

            state, flag = filter_obj.fx(x, eps)

            simobs = filter_obj.hx(state)

            llobs = -logpdf(simobs, mean=obs, cov=filter_obj.R)
            lleps = -logpdf(eps, mean=np.zeros_like(obs), cov=filter_obj.eps_cov)

            if flag:
                return llobs + lleps + penalty

            return llobs + lleps

    if 1 in itype:
        def maintarget(eps, x, mean, cov):

            state, flag = filter_obj.fx(x, eps)

            if flag:
                return -logpdf(state, mean=mean, cov=cov) + penalty

            return -logpdf(state, mean=mean, cov=cov)

    superfflag = False

    for t in range(means[:-1].shape[0]):

        if presmoothing == 'off':
            eps = np.zeros(filter_obj._dim_z)
        elif presmoothing == 'residuals':
            eps = (means[t+1] - filter_obj.fx(means[t],
                                        np.zeros(filter_obj._dim_z))[0]) @ x2eps
        elif presmoothing == 'linear':
            eps = nl.pinv(Ce + T2.T @ Cz @
                          T2) @ T2.T @ Cz @ (filter_obj.Z[t+1] - T1 @ x - T3)
        else:
            raise Exception('[ipas:]'.ljust(15, ' ')+" 'Presmoothing' must be either 'off', residuals' or 'linear'.")

        if 0 in itype:
            res = so_minimize(pretarget, eps, method=method, args=(
                x, filter_obj.Z[t+1]), options=min_options)
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

        x, fflag = filter_obj.fx(x, noise=eps)
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
