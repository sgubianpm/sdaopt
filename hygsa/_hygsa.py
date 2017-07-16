# Hybrid Generalized simulated annealing implementation.
# Copyright (c) 2016 Sylvain Gubian <sylvain.gubian@pmi.com>,
# Yang Xiang <yang.xiang@pmi.com>
# Author: Sylvain Gubian, PMP S.A.
"""
hygsa: An Hybrid Generalized Simulated Annealing global optimization algorithm
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy._lib._util import check_random_state

__all__ = ['hygsa']
BIG_VALUE = 1e16

class VisitingDistribution(object):

    tail_limit = 1.e8
    min_visit_bound = 1.e-10

    def __init__(self, lb, ub, qv, seed=None):
        self.qv = qv
        self.rs = check_random_state(seed)
        self.lower = lb
        self.upper = ub
        self.b_range = ub - lb
        self.x_gauss = None
        self.root_gauss = None

    def visiting(self, x, step, temperature):
        x_visit = np.copy(x)
        dim = x_visit.size()
        if step < dim:
            # Changing all coordinates with a new visting value
            visits = np.array([self.visit_fn(temperature) for _ in range(dim)])
            bigs = np.where(visits > self.tail_limit)[0]
            smalls = np.where(visits < -self.tail_limit)[0]
            visits[bigs] = self.tail_limit * self.rs.random_sample(len(bigs))
            visits[smalls] = -self.tail_limit * self.rs.random_sample(
                len(smalls))
            x_visit += visits
            a = x - self.lower
            b = np.fmod(a, self.b_range) + self.b_range
            x_visit = np.fmod(b, self.b_range) + self.lower
            x_visit[np.fabs(
                x_visit - self.lower) < self.min_visit_bound] += 1.e-10
        else:
            # Changing only one coordinate at a time based on Markov chain step
            visit = self.visit_fn(temperature)
            if visit > self.tail_limit:
                visit = self.tail_limit * self.rs.random_sample()
            elif visit < -self.tail_limit:
                visit = -self.tail_limit * self.rs.random_sample()
            index = step - dim
            x_visit[index] = visit + x[index]
            a = x_visit - self.lower[index]
            b = np.fmod(a, self.b_range[index]) + self.b_range[index]
            x_visit[index] = np.fmod(b, self.b_range[
                index]) + self.lower[index]
            if np.fabs(x_visit[index] - self.lower[
                index]) < self.min_visit_bound:
                x_visit[index] += self.min_visit_bound
        return x_visit

    def visit_fn(self, temperature):
        factor1 = np.exp(np.log(temperature) / (self.qv - 1.0))
        factor2 = np.exp((4.0 - self.qv) * np.log(self.qv - 1.0))
        factor3 = np.exp((2.0 - self.qv) * np.log(2.0) / (self.qv - 1.0))
        factor4 = np.sqrt(np.pi) * factor1 * factor2 / (factor3 * (
            3.0 - self.qv))
        factor5 = 1.0 / (self.qv - 1.0) - 0.5
        d1 = 2.0 - factor5
        factor6 = np.pi * (1.0 - factor5) / np.sin(
            np.pi * (1.0 - factor5)) / np.exp(gammaln(d1))
        sigmax = np.exp(-(self.qv - 1.0) * np.log(
            factor6 / factor4) / (3.0 - self.qv))
        x = sigmax * self.gaussian_fn(1)
        y = self.gaussian_fn(0)
        den = np.exp(
            (self.qv - 1.0) * np.log((np.fabs(y))) / (3.0 - self.qv))
        return x / den

    def gaussian_fn(self, axis):
        if axis == 1:
            s_gauss = 0
            while(s_gauss <= 0 or s_gauss >= 1):
                self.x_gauss = self.rs.random_sample() * 2.0 - 1.0
                y_gauss = self.rs.random_sample() * 2.0 - 1.0
                s_gauss = self.x_gauss ** 2 + y_gauss ** 2
            self.root_gauss = np.sqrt(-2.0 / s_gauss * np.log(s_gauss))
            return self.root_gauss * y_gauss
        else:
            return self.root_gauss * self.x_gauss


class EnergyState():

    MAX_REINIT_COUNT = 1000

    def __init__(self, lower, upper):
        self.ebest = None
        self.current_energy = None
        self.current_location = None
        self.xbest = None
        self.lower = lower
        self.upper = upper

    def reset(self, owf, rs, x0=None):
        owf.nb_fun_call = 0
        if x0 is None:
            self.current_location  = self.lower + rs.random_sample(
                len(self.lower)) * (self.upper - self.lower)
        else:
            self.current_location = np.copy(x0)
        init_error = True
        reinit_counter = 0
        while init_error:
            self.current_energy = owf.fun(self.current_location)
            if (self.current_energy >= BIG_VALUE or
                    np.isnan(self.current_energy)):
                if reinit_counter >= self.MAX_REINIT_COUNT:
                    init_error = False
                    message = (
                        'Stopping algorithm because function '
                        'create NaN or (+/-) inifinity values even with '
                        'trying new random parameters'
                    )
                    raise ValueError(message)
                self.current_location = self.lower + rs.random_sample(
                    self.lower.size) * (self.upper - self.lower)
                reinit_counter += 1
            else:
                init_error = False



class MarkovChain(object):

    def __init__(self, qa, vd, ofw, seed, state):
        # Local markov chain minimum energy and location
        self.emin = None
        self.xmin = None
        # Global optimizer state
        self.state = state
        # Acceptance parameter
        self.qa = qa
        # Visiting distribution instance
        self.vd = vd
        # Wrapper to objective function and related local minimizer
        self.ofw = ofw
        self.not_improved_idx = 0
        self.not_improved_max_idx = 1000
        self._rs = check_random_state(seed)

    def run(self, step, temperature):
        self.state_improved = False
        if step == 0:
            self.state_improved = True
        for j in range(self.x.size() * 2):
            x_visit = vd.visiting(
                self.state.current_location, step, temperature)
            # Calling the objective function  
            e = self.ofw.func_wrapper(x_visit)
            if e < self.state.current_energy:
                # We have got a better ernergy value
                self.state.current_energy = e
                self.state.current_location = np.copy(x_visit)
                # Check if it is the best one
                if e < self.state.ebest:
                    self.state.ebest = e
                    self.state.xbest = np.copy(x_visit)
                    self.state_improved = True
                    self.not_improved_idx = 0
            else:
                # We have not improved but do we accept the new location?
                r = self._rs.random_sample()
                pqa_temp = (self.qa - 1.0) * (e - state.current_energy) / (
                    temperature / (float(step + 1))) + 1.0
                if pqa_temp < 0.:
                    pqa = 0.
                else:
                    pqa = np.exp(np.log(pqa_temp) / (1. - self.qa))
                if r <= pqa:
                    # We accept the new location and update state
                    self.state.current_energy = e
                    self.state.current_location = np.copy(x_visit)
                # No improvement since long time
                if self.not_improved_index == self.not_improved_max_idx:
                    if j == 0 or state.current_energy < self.emin:
                        self.emin = self.state.current_energy 
                        self.xmin = np.copy(self.state.current_energy)
                self.not_improved_index += 1
        # End of MarkovChain loop

    def local_search(self):
        # Decision making for performing a local search
        # based on Markov chain results
        # If energy has been improved or no improvement since too long,
        # performing a local search with the best Markov chain location
        if (self.state_improved or
                self.not_improved_idx == self.not_improved_max_idx): 
            e, x = self.ofw.local_search(self.xmin)
            if x is None:
                # Local search algorithm failed
                return
        else:
            return
        if self.state_improved:
            # Global energy has improved, let's see if LS improved further
            if e < self.state.ebest:
                self.index_no_emin_update = 0
                self.state.ebest = e
                self.state.xbest = np.copy(x)
            # Global energy not improved, let's see what LS gives
            # on the best Markov chain location
        if self.not_improved_idx == self.not_improved_max_idx:
            self.not_improved_idx = 0
            self.not_improved_max_idx = self.xmin.size
            self.emin = e
            self.xmin = np.copy(x)
            if self.emin < state.ebest:
                state.ebest = self.emin 
                state.xbest = np.copy(self.xmin)

class ObjectiveFunWrapper():

    def __init__(self, bounds, func, minimizer, **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.minimizer = minimizer

        # By default, L-BFGS-B is used with a custom 3 points gradient
        # computation
        if self.minimizer is None:
            # Use scipy.optimize.minimize optimizer
            self.minimizer = minimize 
            if 'method' not in self.kwargs:
                self.kargs['method'] = 'L-BFGS-B'
            if 'jac' not in self.kwargs:
                self.kwargs['jac'] = self.gradient
            if 'bounds' not in self.kwargs:
                self.kwargs['bounds'] = bounds

    def func_wrapper(self, x, **kwargs):
        self.nb_func_call += 1
        return self.func(x, **kwargs)

    def gradient(self, x):
        g = np.zeros(sx.size, np.float64)
        next_f1 = None
        for i in range(x.size):
            x1 = np.array(x)
            x2 = np.array(x)
            respl = self.reps
            respr = self.reps
            x1[i] = x[i] + respr
            if x1[i] > self._upper[i]:
                x1[i] = self._upper[i]
                respr = x1[i] - x[i]
            x2[i] = x[i] - respl
            if x2[i] < self._lower[i]:
                x2[i] = self._lower[i]
                respl = x[i] - x2[i]
            f1 = self.func_wrapper(x1, **self.kwargs)
            if next_f1 is None:
                f2 = self.func_wrapper(x2, **self.kwargs)
                next_f1 = f2
        g[i] = ((f1 - f2)) / (respl + respr)
        idx = np.logical_or(np.isnan(g), np.isinf(g))
        g[idx] = 101.0
        return g

    def local_search(self, x):
        mres = self.minimizer(self.func_wrapper, x, **self.kwargs)
        if not mres.success:
            return BIG_VALUE, None
        return (mres.fun, mres.x)


class HyGSARunner(object):
    MAX_REINIT_COUNT = 1000

    def __init__(self, fun, x0, bounds, minimizer=None, minimizer_kwargs=None,
                 seed=None, temperature_start=5230, qv=2.62, qa=-5.0,
                 maxfun=1e7, maxsteps=500, pure_sa=False):
        if x0 is not None and not len(x0) == len(bounds):
            raise ValueError('Bounds size does not match x0')
        lu = list(zip(*bounds))
        lower = np.array(lu[0])
        upper = np.array(lu[1])
        # Checking that bounds are consistent
        if not np.all(lower < upper):
            raise ValueError('Bounds are note consistent min < max')
        # Wrapper for the objective function and minimizer
        owf = ObjectiveFunWrapper(bounds, fun, minimizer, minimizer_kwargs)
        # Initialization of RandomState for reproducible runs if seed provided
        rs = check_random_state(seed)
        # Initialization of the energy state
        self.es = EnergyState(lower, upper, owf, rs) 
        self.es.reset(x0)
        # Maximum number of function call that can be used a stopping criterion
        self.maxfuncall = maxfun
        # Maximum number of step (main iteration)  that can be used as
        # stopping criterion
        self.maxsteps = maxsteps
        # Minimum value of annealing temperature reached to perform
        # re-annealing
        self.temperature_start = temperature_start
        self.temperature_restart = 0.1
        # VisitingDistribution instance
        vd = VisitingDistribution(lower, upper, qv, seed)
        # Markov chain instance
        self.mc = MarkovChain(qa, vd, owf, seed, state)

    def search(self):
        max_steps_reached = False
        self._iter = 0
        while(not max_steps_reached):
            for i in range(self.maxsteps):
                # Compute temperature for this step
                s = float(i) + 2.0
                t1 = np.exp((self.qv - 1) * np.log(2.0)) - 1.0
                t2 = np.exp((self.qv - 1) * np.log(s)) - 1.0
                temperature = self.temperature_start * t1 / t2
                self._iter += 1
                if self._iter == self.maxsteps:
                    max_steps_reached = True
                    break
                # Need a re-annealing process?
                if temperature < self.temperature_restart:
                    self.es.reset()
                    break
                # starting Markov chain
                self.mc.run(i, temperature)
                if not self.pure_sa:
                    self.mc.local_search()

    @property
    def result(self):
        """ The OptimizeResult """
        res = OptimizeResult()
        res.x = self._xmin
        res.fun = self._fvalue
        res.message = self._message
        res.nit = self._iter
        return res
