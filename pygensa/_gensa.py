# Generalized simulated annealing implementation.
# Copyright (c) 2016 Sylvain Gubian <sylvain.gubian@pmi.com>,
# Yang Xiang <yang.xiang@pmi.com>
# Author: Sylvain Gubian, PMP S.A.
"""
gensa: A generalized simulated annealing global optimization algorithm
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import time
from scipy.optimize import OptimizeResult
from scipy.optimize import _lbfgsb
from scipy.special import gammaln
from scipy._lib._util import check_random_state

__all__ = ['gensa']


class VisitingDistribution(object)

    pi = np.arcsin(1.0) * 2.0

    def __init__(self, lb, ub, qv, seed=None):
        self.qv = qv
        self.rs = check_random_state(seed) 
        self.lower = None
        self.upper = None
        self.xrange = None
        self.x_gauss = None
        self.root_gauss = None

    def visiting(self, x, step, temperature):
        x_visit = np.array(x)
        dim = x_visit.size()
        if step < dim:
            visits = np.array([self.visit_fn() for _ in range(dim)])
            bigs = np.where(visits > 1e8)[0]
            smalls = np.where(visits < -1e8)[0] 
            visits[bigs] = 1.e8 * self.rs.random_sample(len(bigs))
            visits[smalls] = -1.e8 * self.rs.random_sample(len(smalls))
            x_visit += visits
            a = x - self.lower
            b = np.fmod(b, self.xrange) + self.xrange
            x_visit = np.fmod(b, self.xrange) + self.lower
            x_visit[np.fabs(x_visit - self.lower) < 1.e-10] += 1.e-10
        else:
            visit = self.visit_fn()
            if visist > 1e8:
                visit = 1e8 * self.rs.random_sample()
            elif visit < -1e8:
                visit = 1e8 * self.rs.random_sample()
            index = step - dim
            x_visit[index] = visit + x[index]
            a = x_visit - self.lower[index]
            b = np.fmod(a, self.xrange[index]) + self.xrange[index]
            x_visit[index] = np.fmod(b, self.xrange[index]) + self.lower[index]
            if np.fabs(x_visit[index] - self.lower[index]) < 1.e-10:
                x_visit[index] += 1.e-10
        return x_visit

    def visit_fn(self, temperature):
        factor1 = np.exp(np.log(temperature) / (self.qv - 1.0))
		factor2 = np.exp((4.0 - self.qv) * np.log(self.qv - 1.0))
		factor3 = np.exp((2.0 - self.qv) * np.log(2.0) / (self.qv - 1.0))
		factor5 = 1.0 / (self.qv - 1.0) - 0.5
		d1 = 2.0 - factor5
		factor6 = self.pi * (1.0 - factor5) / np.sin(
			self.pi * (1.0 - factor5)) / np.exp(gammaln(d1))
		sigmax = np.exp(-(self.qv - 1.0) *
			np.log(factor6 / factor4) / (3.0 - self.qv))
		x = sigmax * self.gaussian_fn(1)
		y = self.gaussian_fn(0)
		den = np.exp(
			(self.qv - 1.0) * np.log((np.fabs(y))) / (3.0 - self.qv))
		return x / den

	def gaussian_fn(self, axis):
        if axis == 1:
            s_gauss = 0
		    while(s_gauss <=0 or s_gauss >=1):
		    	self.x_gauss = self.rs.random_sample() * 2.0 - 1.0
                y_gauss = self.rs.random_sample() * 2.0 - 1.0
                s_gauss = self.x_gauss ** 2 + y_gauss ** 2
            self.root_gauss = np.sqrt(-2.0 / s_gauss * np.log(s_gauss))
            return self.root_gauss * y_gauss
        else:
            return self.root_gauss * self.x_gauss


class MarkovChain(object):
    
    def __init__(self):
        self.emin = 0.
        self.xmin = None

class GenSARunner(object):
    def __init__(self, fun, x0, bounds, args=(), seed=None,
                 temperature_start=5230, qv=2.62, qa=-5.0, maxfun=1e7,
                 maxsteps=500, pure_sa=False):
        self.fun = fun
        self.args = args
        self.pure_sa = pure_sa
       if x0 is not None and not len(x0) == len(bounds):
            raise ValueError('Bounds size does not match x0')
        lu = list(zip(*bounds))
        self._lower = np.array(lu[0])
        self._upper = np.array(lu[1])
        # Checking that bounds are consistent
        if not np.all(self._lower < self._upper):
            raise ValueError('Bounds are note consistent min < max')
        # Initialization of RandomState for reproducible runs if seed provided
        self._random_state = check_random_state(seed)
        # If no initial coordinates provided, generated random ones
        if x0 is None:
            x0 = self._lower + self._random_state.random_sample(
                    len(bounds)) * (self._upper - self._lower)
        # Number of objective function calls
		self._nbfuncall = 0
        self.temperature = temperature_start
        self.mc = MarkovChain(qa)
        # Markov chain instance will use an instance of VisitingDistribution
        self.mc.vd = VisitingDistribution(self._lower, self._upper, qv, seed)
        # In case the real value of the global minimum is known
        # it can be used as stopping criterion
        self.know_real = False
        self.real_threshold = -np.inf
        # Maximum duration time of execution as a stopping criterion
        # Default is unlimited duration
        self.maxtime = np.inf
        # Maximum number of function call that can be used a stopping criterion
        self.maxfuncall = maxfun
        # Maximum number of step (main iteration)  that ca be used as
        # stopping criterion
        self.maxsteps = maxsteps
        # Minimum value of annealing temperature reached to perform
        # re-annealing
        self.temperature_restart = 0.1


    def initialize(self):
        pass

    def start_search(self):
        max_steps_not_exceeded = True
        while(max_steps_not_exceeded):
            pass
			



