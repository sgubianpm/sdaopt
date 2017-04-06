# Generalized simulated annealing implementation.
# Copyright (c) 2016 Sylvain Gubian <sylvain.gubian@pmi.com>,
# Yang Xiang <yang.xiang@pmi.com>
# Author: Sylvain Gubian, PMP S.A.
"""
gensa: A generalized simulated annealing global optimization algorithm
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import _lbfgsb
from scipy.special import gammaln
from scipy._lib._util import check_random_state

__all__ = ['gensa']


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
            visits = np.array([self.visit_fn() for _ in range(dim)])
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
            visit = self.visit_fn()
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


class MarkovChain(object):

    def __init__(self, qa, vd, ofw, seed):
        self.emin = None
        self.xmin = None
        self.qa = qa
        self.vd = vd
        self.ofw = ofw
        self.emin_unchanged = True
        self.index_no_emin_update = 0
        self.index_tol_emin_update = 1000
        self.current_energy = None
        self._rs = check_random_state(seed)

    def run(self, x, step, temperature):
        self.index_no_emin_update += 1
        for j in range(self.xmin.size()):
            if j == 0:
                self.emin_unchanged = True
            if step == 0 and j == 0:
                self.emin_unchanged = False
            x_visit = vd.visiting(x, step, temperature)
            e = self.ofw.func(x_visit)
            if e < self.current_energy:
                # We have got a better ernergy value
                self.current_energy = e
                if e < self.emin:
                    self.emin = e
                    self.xmin = np.copy(x_visit)
                    self.emin_unchanged = False
                    self.index_no_emin_update = 0
                    # Updating x with the new location
                    x = np.copy(x_visit)
            else:
                # We have not improved but do we accept the new location?
                r = self._rs.random_sample()
                pqa1 = (self.qa - 1.0) * (e - self.current_energy) / (
                    temperature / (float(step + 1))) + 1.0
                if pqa1 < 0.:
                    pqa = 0.
                else:
                    pqa = np.exp(np.log(pqa1) / (1. - self.qa))
                if r <= pqa:
                    # We accept the new location and assign the current x
                    x = np.copy(x_visit)
                    self.current_energy = e
                if self.index_no_emin_update >= self.index_tol_emin_update:
                    if j == 0:
                        self.emin_markov = self.current_energy
                        self.xmin_markov = np.copy(x)
                    else:
                        if self.current_energy < self.emin_markov:
                            self.emini_markov = self.current_energy
                            self.xmin_markov = np.copy(x)
        # End of MarkovChain loop
        # Decision making for performing a local search
        # based on Markov chain results
        if not self.emin_unchanged:
            e, x = self.ofw.local_search(self.xmin)
            if e < self.emini:
                self.xmin = x
                self.emin = e
                self.index_no_emin_update = 0
        if self.index_no_emin_update >= (
            self.index_tol_emin_update - 1) and not self.pure_sa:
            self.emin_markov, self.xmin_markov = self.ofw.local_search(
                self.xmin_markov)
            self.index_no_emin_update = 0
            self.index_tol_emin_update = x.size
            if self.emin_markov < self.emin:
                self.emin = np.copy(self.emin_markov)
                self.xmin = np.copy(self.xmin_markov)
        return x

class ObjectiveFunctionWrapper():

    def __init__(self, bounds, func, args=(), minimizer=None, **kwargs):
        # In case the real value of the global minimum is known
        # it can be used as stopping criterion
        self.know_real = False
        self.real_threshold = -np.inf
        self.func = func
        self.func_args = args
        self.minimizer = minimizer
        self.minimizer_args = kwargs

        if self.minimizer is None:
            minimizer = optimize.minimize
            self.minimizer_args = {
                "method": "L-BFGS-B",
                "jac": self.gradient,
                "bounds": bounds,
            }

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
            f1 = self.func(x1, self.func_args)
            if next_f1 is None:
                f2 = self.func(x2, self.func_args)
                next_f1 = f2
        g[i] = ((f1 - f2)) / (respl + respr)
        idx = np.logical_or(np.isnan(g), np.isinf(g))
        g[idx] = 101.0
        return g

    def local_search(self, x):
        mres = self.minimizer(self.func, x, **self.minimizer.args)
        if not mres.success:
            return None
        return (mres.fun, mres.x)



class GenSARunner(object):

    def __init__(self, fun, x0, bounds, args=(), seed=None,
                 temperature_start=5230, qv=2.62, qa=-5.0, maxfun=1e7,
                 maxsteps=500):
        self.owf = ObjFuncWrapper(fun, args)
        if x0 is not None and not len(x0) == len(bounds):
            raise ValueError('Bounds size does not match x0')
        lu = list(zip(*bounds))
        self._lower = np.array(lu[0])
        self._upper = np.array(lu[1])
        # Checking that bounds are consistent
        if not np.all(self._lower < self._upper):
            raise ValueError('Bounds are note consistent min < max')
        # Initialization of RandomState for reproducible runs if seed provided
        self._rs = check_random_state(seed)
        # If no initial coordinates provided, generated random ones
        if x0 is None:
            x0 = self._lower + self._rs.random_sample(
                    len(bounds)) * (self._upper - self._lower)
        # Current location (parameter value)
        self._x = np.copy(x0)
        # Initial energy value
        self._einit = None
        # Maximum number of function call that can be used a stopping criterion
        self.maxfuncall = maxfun
        # Maximum number of step (main iteration)  that ca be used as
        # stopping criterion
        self.maxsteps = maxsteps
        # Minimum value of annealing temperature reached to perform
        # re-annealing
        self._temperature_start = temperature_start
        self.temperature_restart = 0.1
        # Markov chain instance
        vd = VisitingDistribution(self._lower, self._upper, qv, seed)
        self.mc = MarkovChain(qa, vd, self.owf)

    def initialize(self, x0):
        init_error = True
        reinit_counter = 0
        self.owl.nb_fn_call = 0
        while init_error:
            self._einit = self.owf.fun(self._x)
            if self._einit >= self.BIG_VALUE:
                if reinit_counter >= self.MAX_REINIT_COUNT:
                    init_error = False
                    self._message = [(
                        'Stopping algorithm because function '
                        'create NaN or (+/-) inifinity values even with '
                        'trying new random parameters'
                    )]
                    raise ValueError(self._message[0])
                self._x = self._lower + self._rs.random_sample(
                    self._xinit.size) * (self._upper - self._lower)
                reinit_counter += 1
            else:
                init_error = False

    def search(self):
        max_steps_not_exceeded = True
        while(max_steps_not_exceeded):
            for i in range(self.maxsteps):
                # Compute temperature for this step
                s = float(i) + 2.0
                t1 = np.exp((self.qv - 1) * np.log(2.0)) - 1.0
                t2 = np.exp((self.qv - 1) * np.log(s)) - 1.0
                temperature = self.temperature_start * t1 / t2
                self._step_record += 1
                if self._step_record == self.maxsteps:
                    max_steps_not_exceeded = False
                    break
                # Need a re-annealing process?
                if temperature < self._temperature_restart:
                    break
                # starting Markov chain
                self._x = self.mc.run(self._x, i, temperature)


    def local_search(self):
        pass

    def gradient(self):
        pass

    def ernergy(self):
        pass

    @property
    def result(self):
        """ The OptimizeResult """
        res = OptimizeResult()
        res.x = self._xmin
        res.fun = self._fvalue
        res.message = self._message
        res.nit = self._step_record
        return res


