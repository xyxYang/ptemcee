# -*- coding: utf-8 -*- 
# @Time : 2025/5/21 10:55 
# @Author : yangyuxin
# @File : moves.py

import abc
import numpy as np
from collections import namedtuple

__all__ = ["Model", "Move", "StretchMove", "MHMove", "GaussianMove"]

Model = namedtuple(
    "Model", ("evaluate", "tempered_likelihood", "random")
)


class Move(object):
    @abc.abstractmethod
    def propose(self, model, x, logP, logl):
        """
        Use the move to generate a proposal and compute the acceptance
        :param model: Model class includes evaluate, tempered_likelihood and random to generate log likelihood
                      and log priors. Evaluate function returns log likelihood and log priors. Tempered_likelihood
                      function generates tempered log likelihood by log likelihood result.
        :param x: The samples of old state, with shape of (ntemps, nwalkers, ndim).
        :param logP: The log probability of old state. (consider the effect of temper)
        :param logl: The log likelihood of old state.

        :return: x, logP, logl and jumps_accepted of new state.
        """


class StretchMove(Move):
    def __init__(self, scale_factor=2):
        self._scale_factor = scale_factor

    def propose(self, model, x, logP, logl):
        ntemps, nwalkers, ndim = x.shape
        jumps_accepted = np.zeros((ntemps, nwalkers))
        w = nwalkers // 2
        d = ndim
        t = ntemps
        loga = np.log(self._scale_factor)

        for j in [0, 1]:
            # Get positions of walkers to be updated and walker to be sampled.
            j_update = j
            j_sample = (j + 1) % 2
            x_update = x[:, j_update::2, :]
            x_sample = x[:, j_sample::2, :]

            z = np.exp(model.random.uniform(low=-loga, high=loga, size=(t, w)))
            y = np.empty((t, w, d))
            for k in range(t):
                js = model.random.randint(0, high=w, size=w)
                y[k, :, :] = (x_sample[k, js, :] +
                              z[k, :].reshape((w, 1)) *
                              (x_update[k, :, :] - x_sample[k, js, :]))

            y_logl, y_logp = model.evaluate(y)
            y_logP = model.tempered_likelihood(y_logl) + y_logp

            logp_accept = d * np.log(z) + y_logP - logP[:, j_update::2]
            logr = np.log(model.random.uniform(low=0, high=1, size=(t, w)))

            accepts = logr < logp_accept
            accepts = accepts.flatten()

            x_update.reshape((-1, d))[accepts, :] = y.reshape((-1, d))[accepts, :]
            logP[:, j_update::2].reshape((-1,))[accepts] = y_logP.reshape((-1,))[accepts]
            logl[:, j_update::2].reshape((-1,))[accepts] = y_logl.reshape((-1,))[accepts]

            jumps_accepted[:, j_update::2] = accepts.reshape((t, w))
        return x, logP, logl, jumps_accepted


class MHMove(Move):
    def __init__(self, proposal_function):
        self.get_proposal = proposal_function

    def propose(self, model, x, logP, logl):
        ntemps, nwalkers, ndim = x.shape

        # generate new proposal
        y, factors = self.get_proposal(x, model.random)
        y_logl, y_logp = model.evaluate(y)
        y_logP = model.tempered_likelihood(y_logl) + y_logp
        logp_diff = y_logP - logP + factors

        # calculate witch one need to be accepted
        logr = np.log(model.random.uniform(low=0, high=1, size=(ntemps, nwalkers)))
        accepts = logr < logp_diff
        accepts = accepts.flatten()

        x.reshape((-1, ndim))[accepts, :] = y.reshape((-1, ndim))[accepts, :]
        logP.reshape((-1,))[accepts] = y_logP.reshape((-1,))[accepts]
        logl.reshape((-1,))[accepts] = y_logl.reshape((-1,))[accepts]

        return x, logP, logl, accepts.reshape((ntemps, nwalkers))


class GaussianMove(MHMove):
    def __init__(self, cov):
        self._cov = cov
        super(GaussianMove, self).__init__(self.get_proposal)

    def get_proposal(self, x, random):
        temps, nwalkers, ndim = x.shape
        if len(self._cov.shape) == 1:
            y = x + self._cov * random.randn(*x.shape)
        elif len(self._cov.shape) == 2:
            y = x + random.multivariate_normal(np.zeros(len(self._cov)), self._cov)
        return y, np.zeros((temps, nwalkers))
