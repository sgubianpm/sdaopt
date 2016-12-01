##############################################################################
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
##############################################################################
# -*- coding: utf-8 > -*-
__author__ = "Sylvain Gubian"
__copyright__ = "Copyright 2016, PMP SA"
__license__ = "GPL2.0"
__email__ = "Sylvain.Gubian@pmi.com"

import sys
import os
import time
import logging
import pickle
import csv
import math
import multiprocessing
import inspect
import numpy as np
from scipy import optimize
from pyswarm import pso
import go_benchmark_functions as gbf

logger = logging.getLogger(__name__)

DEFAULT_TOL = 1.e-8
MAX_FN_CALL = 1e6
LS_MAX_CALL = 1e4
MAX_IT = int(1e6)
if 'USE_CLUSTER' in os.environ:
    assert('NB_CORES' in os.environ)
    assert('SECTION_NUM' in os.environ)
    NB_CORES_AVAILABLES = int(os.environ['NB_CORES'])
else:
    NB_CORES_AVAILABLES = multiprocessing.cpu_count()
if NB_CORES_AVAILABLES < 1:
    NB_CORES_AVAILABLES = 1

if 'MULTI_DIM' in os.environ:
    MULTI_DIM = True
else:
    MULTI_DIM = False
DIMENSIONS = [5, 10, 20, 30, 40, 50]

METRICS = [
        'success',
        'ncall',
        'fvalue',
        'time',
        'ncall_max',
        ]


N_DIM_FUNC_SELECTION = [
    'Ackley01', 'Exponential', 'Griewank', 'Rastrigin', 'Rosenbrock',
    'Schwefel01',
    ]


class MyBounds(object):
    def __init__(self, xmax, xmin):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


class BenchUnit(object):
    def __init__(self, nbruns, name, algo):
        self._name = name
        self._algo = algo
        self._nbruns = nbruns
        self._values = dict.fromkeys(METRICS)
        for k, v in self._values.items():
            self._values[k] = np.empty(nbruns)
            self._values[k][:] = np.NAN

    def __str__(self):
       return '{0}-{1}'.format(self._name, self._algo,)

    def update(self, key, index, value):
        self._values[key][index] = value

    def replicate(self):
        """ Only values of the first run are necessary
        for non stochastic approach (like brute force)
        Then this method can be called to copy the first
        value to all the runs
        """
        for k in self._values:
            self._values[k][1:] = [self._values[k][0]] * (self._nbruns - 1)

    def values(self):
        return self._values

    @property
    def name(self):
        return self._name

    @property
    def algo(self):
        return self._algo

    @property
    def best(self):
        v = self._values['ncall']
        v = v[np.where(self.success)]
        if not v.size:
            return np.inf
        #return int(np.amin(v))
        return np.round(np.amin(v), 6)

    @property
    def worst(self):
        v = self._values['ncall']
        v = v[np.where(self.success)]
        if not v.size:
            return np.inf
        #return int(np.amax(v))
        return np.round(np.amax(v), 6)

    @property
    def mean(self):
        v = self._values['ncall']
        v = v[np.where(self.success)]
        if not v.size:
            return np.inf
        return np.round(np.mean(v), 6)

    @property
    def med(self):
        v = self._values['ncall']
        v = v[np.where(self.success)]
        if not v.size:
            return np.inf
        return np.round(np.median(v), 6)

    @property
    def std(self):
        v = self._values['ncall']
        v = v[np.where(self.success)]
        if not v.size:
            return np.inf
        return np.round(np.std(v), 6)

    @property
    def lowest(self):
        v = self._values['fvalue']
        v = v[np.where(self.success)]
        if not v.size:
            return np.inf
        return np.round(np.min(v), 8)

    @property
    def highest(self):
        v = self._values['fvalue']
        v = v[np.where(self.success)]
        if not v.size:
            return np.inf
        return np.round(np.max(v), 2)

    @property
    def success(self):
        v = self._values['success']
        return v

    @property
    def x(self):
        v = self._values['xvalues']
        v = v[np.where(self.success)]
        if not v.size:
            return np.inf
        return np.round(np.mean(v, 0), 8)

    @property
    def time(self):
        v = self._values['time']
        v = v[np.where(self.success)]
        if not v.size:
            return np.inf
        return np.round(np.mean(v), 6)

    @property
    def filename(self):
        return '{0}-{1}-{2}.data'.format(
            self._name, self._algo, self._nbruns,)

    def write(self, folder=os.getcwd()):
        if not os.path.exists(folder):
            os.mkdir(folder)
        filename = os.path.join(folder, self.filename)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


class BenchStore(object):
    @staticmethod
    def report(folder, kind="raw", path=None):
        """ By default, report creates a CSV file with the benchmark results
        for each function and method used
        If kind is ``raw`` no file is generated, a dict is returned with
        Method/Function/values
        If kind is ``csv``, a csv file is generated with the results
        If kind is ``rst``, a rst file is generated with the table in rst
        format.
        """
        files = os.listdir(folder)
        files = [x for x in files if x.endswith('.data')]
        files = sorted(files)
        if kind == 'csv':
            with open(path, 'wb') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(['Function name', 'Algorithm',
                    'Success Rate',
                    'Best', 'Average','Worst', 'Std', 'fvalue (mean)',
                    'Mean time (ms)'])
                for f in files:
                    with open(os.path.join(folder, f), 'rb') as fh:
                        bu = pickle.load(fh)
                        logger.info('Processing benchunit: {0}'.format(bu))
                        et = bu.time
                        if et < 0:
                            et = np.inf
                        else:
                            et *= 1000
                        success_rate = np.round(np.where(bu.success)[0].size *\
                            100 / bu.success.size, 1)
                        csvwriter.writerow([bu.name, bu.algo,
                            "{0}%".format(success_rate),
                            bu.best, bu.mean, bu.worst,
                            '{0} {1}'.format('(+/-)', bu.std), bu.lowest,
                            round(et,6)])
        elif kind == 'rst':
            table = []
            counter = 3
            for f in files:
                with open(os.path.join(folder, f), 'rb') as fh:
                    bu = pickle.load(fh)
                    logger.info('Processing benchunit: {0}'.format(bu))
                    et = bu.time
                    if et < 0:
                        et = np.inf
                    else:
                        et *= 1000
                    success_rate = np.round(np.where(bu.success)[0].size *\
                        100 / bu.success.size, 1)
                    if 'DifferentialEvolution' in bu.algo:
                        algo = 'DE'
                    elif bu.algo == 'BasinHopping':
                        algo = 'BH'
                    else:
                        algo = bu.algo
                    if counter % 3 == 0:
                        name = bu.name
                    else:
                        name = ""
                    table.append([name, algo,
                        "{0}%".format(success_rate),
                        "{0}".format(bu.best),
                        "{0}".format(bu.med),
                        "{0}".format(bu.worst), '{0} {1:.2f}'.format('+/-',
                        bu.std),])
                counter += 1
            with open(path, 'w') as outf:
                for line in table:
                    outf.write('| ')
                    outf.write(' | '.join(line))
                    outf.write(' |')
                    outf.write('\n')
        else:
            d = {}
            for f in files:
                with open(os.path.join(folder, f), 'rb') as fh:
                    bu = pickle.load(fh)
                    logger.info('Processing benchunit: {0}'.format(bu))
                    if bu.algo not in d:
                        d[bu.algo] = {}
                    if bu.name not in d[bu.algo]:
                        d[bu.algo][bu.name] = bu.values()
            return d




class Job(object):
    def __init__(self, name, klass, dim=None):
        self.name = name
        self.process = None
        self._status = 'INIT'
        self._klass = klass
        self.dim = dim

    def __str__(self):
        if self.dim is None:
            return 'JOB-{0}: {1}'.format(self.index,
                    self.name,)
        else:
            return 'JOB-{0}: {1} (DIM: {2})'.format(self.index,
                    self.name, self.dim)

    @property
    def status(self):
        if self.process is None:
            return self._status
        if self._status == 'STARTED':
            st = self.process.is_alive()
            if st:
                return 'RUNNING'
            else:
                return 'FINISHED'

    def start(self, func):
        if self.process is None:
            self.process = multiprocessing.Process(
                target=func, args=(self.name, self._klass, self.dim))
        logger.info('Starting job {0}'.format(self.name))
        self.process.start()
        self._status = 'STARTED'


class Benchmarker(object):
    def __init__(self, nbruns, folder):
        self.algorithms = [
            GenSAOptimizer(), BHOptimizer(), DEOptimizer(),
            DERestartOptimizer(), PSOptimizer(), PSOLSOptimizer(),
            PSORestartOptimizer(), PSOLSRestartOptimizer(),
            BFOptimizer()
        ]
        self.nbruns = nbruns
        self.folder = folder
        bench_members = inspect.getmembers(gbf, inspect.isclass)
        self.benchmark_functions = [item for item in bench_members if
                                    issubclass(item[1], gbf.Benchmark)]

    def run(self):
        jobs = {}
        index = 0
        funcs = []
        for name, klass in self.benchmark_functions:
            try:
                k = klass()
            except TypeError:
                k = klass(dimensions=2)
            if MULTI_DIM:
                if k.change_dimensionality and name in N_DIM_FUNC_SELECTION:
                    for dim in DIMENSIONS:
                        logger.info(
                            'Appending function: {0} with dim: {1}'.format(
                                name, dim))
                        funcs.append((name, klass, dim))
            else:
                funcs.append((name, klass, None))
            # Removing Benchmark class that is the mother class
            funcs = [x for x in funcs if x[0] != 'Benchmark']
        logger.info('Nb functions to process: {}'.format(len(funcs)))

        if 'USE_CLUSTER' in os.environ:
            start_idx = int(os.environ['SECTION_NUM']) * int(
                    os.environ['NB_CORES'])
            end_idx = start_idx + int(os.environ['NB_CORES']) - 1
            if end_idx > len(funcs):
                end_idx = len(funcs) - 1
            funcs = funcs[start_idx:(end_idx+1)]
            logger.info('Benchmarking functions: {}'.format(
                [x[0] for x in funcs]))
        for name, klass, dim in funcs:
            #if name == 'Bukin06':
            #    continue
            jobs[index] = Job(name, klass, dim)
            index += 1
        while jobs:
            running = []
            for index, job in jobs.items():
                if job.status == 'FINISHED':
                    del jobs[index]
                if job.status == 'RUNNING':
                    running.append(index)
            freecores = NB_CORES_AVAILABLES - len(running)
            if freecores <= NB_CORES_AVAILABLES:
                non_running = [jobs[
                    x] for x in jobs.keys() if x not in running]
                for i in range(freecores):
                    if i < len(non_running):
                        non_running[i].start(self.bench)
            time.sleep(0.5)


    def bench(self, fname, klass, dim=None):
        '''Benchmarking function. It executes all runs for a specific
        function
        '''
        self._fname = fname
        if dim:
            self._fname = '{0}_{1}'.format(self._fname, dim)
        for algo in self.algorithms:
            bu = BenchUnit(self.nbruns, self._fname, algo.name)
            if os.path.exists(os.path.join(self.folder, bu.filename)):
                logger.info('File {} already existing, skipping...'.format(
                    bu.filename))
                continue
            for i in range(self.nbruns):
                np.random.seed(1234 + i)
                algo.prepare(fname, klass, dim)
                if i > 0 and algo.name == 'BF':
                    logger.info('BRUTE FORCE nbrun > 1, ignoring...')
                    continue
                try:
                    algo.optimize()
                    logger.info(':-(  Func: {0} - Algo: {1} - RUN: {2} -> FAILED after {3} calls'.format(
                        self._fname, algo.name, i, algo.nbcall))
                except Exception as e:
                    if type(e) == OptimumFoundException:
                        logger.info(':-)  Func: {0} - Algo: {1} - RUN: {2} -> FOUND after {3} calls'.format(
                            self._fname, algo.name, i, algo.nbcall))
                        algo._success = True
                    elif type(e) == OptimumNotFoundException:
                        if algo._favor_context:
                            logger.info('Maximum NB CALL reached. Calling LS...')
                            algo.lsearch()
                            self._favor_context = False
                        logger.info(':-(  Func: {0} - Algo: {1} - RUN: {2} -> FAILED after {3} calls'.format(
                            self._fname, algo.name, i, algo.nbcall))
                    else:
                        logger.info(':-(  Func: {0} - Algo: {1} - RUN: {2} -> EXCEPTION RAISED after {3} calls: {4}'.format(
                            self._fname, algo.name, i, algo.nbcall, e))
                        algo._success = False
                    bu.write(self.folder)
                bu.update('success', i, algo.success)
                bu.update('ncall', i, algo.fcall_success)
                bu.update('fvalue', i, algo.fsuccess)
                bu.update('time', i, algo.duration)
                bu.update('ncall_max', i, algo.nbcall)
            if algo.name == 'BF':
                bu.replicate()
            bu.write(self.folder)



class OptimumFoundException(Exception):
    pass


class OptimumNotFoundException(Exception):
    pass


class Algo(object):
    def __init__(self):
        self.name = None
        # Favor context is to allow some algo
        # to run a bit more than MAX_FN_CALL to have
        # the final local search possible (this is for BF and PSO)
        self._favor_context = False

    @property
    def success(self):
        return self._success
    @property
    def fsuccess(self):
        return self._fsuccess

    @property
    def fcall_success(self):
        return self._fcall_success

    @property
    def nbcall(self):
        return self._nbcall

    @property
    def xsuccess(self):
        return self._xsuccess

    @property
    def duration(self):
        return self._hittime - self._starttime

    def prepare(self, fname, klass, dim=None):
        if dim:
            self._k = klass(dimensions=dim)
        else:
            self._k = klass()
        self._fname = fname
        self._xmini = None
        self._fmini = None
        self._favor_context = False
        self._maxcall = MAX_FN_CALL
        self._nbcall = 0
        self._first_hit = True
        self._starttime = time.time()
        self._fcall_success = MAX_FN_CALL
        self._xsuccess = [np.inf] * self._k.N
        self._success = False
        self._fsuccess = np.inf
        self._hittime = np.inf
        self.recording = False
        self._values = []
        self._lower = -np.inf
        self._upper = np.inf
        self._x = None
        # logger.info('Function name: {0} Dimension: {1}'.format(
            # fname, self._k._dimensions))
        self._lower = np.array([x[0] for x in self._k.bounds])
        self._upper = np.array([x[1] for x in self._k.bounds])
        self._xinit = self._lower + np.random.rand(self._k.N) * (
                self._upper - self._lower)

    def optimize(self):
        pass

    def lsearch(self):
        pass

    def _funcwrapped(self, x, **kargs):
        ''' Function to wrap the objective function. This is needed to trace
        when the objective value is reached the first time
        It also records an array with values of the current optimum value
        if required, set recording to True)
        '''
        func = self._k.fun
        res = func(x)
        self._nbcall += 1
        if self._nbcall > self._maxcall:
            if self._favor_context:
                self._maxcall += LS_MAX_CALL
            raise OptimumNotFoundException('NB MAX CALL reached...')
        if self.recording and self._first_hit:
            if len(self.values) > 0:
                if res < self.values[-1]:
                    self.values.append(res)
                else:
                    self.values.append(self.values[-1])
            else:
                self.values.append(res)
        if self._first_hit and res <= self._k.fglob + DEFAULT_TOL:
            self._fcall_success = self._nbcall
            self._fsuccess = res
            self._xsuccess = x
            self._success = True
            self._first_hit = False
            self._hittime = time.time()
            raise OptimumFoundException('FOUND')
        return res


class GenSAOptimizer(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'GenSA'

    def optimize(self):
        Algo.optimize(self)
        ret = optimize.gensa(func=self._funcwrapped, x0=None,
                bounds=zip(self._lower, self._upper), maxiter=MAX_IT,
                pure_sa=False)


class PSOptimizer(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'PSO'

    def optimize(self):
        xopt, fopt = pso(self._funcwrapped, self._lower, self._upper,
                maxiter=MAX_IT)

class PSORestartOptimizer(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'PSO-R'

    def optimize(self):
        while (self._nbcall < MAX_FN_CALL):
            xopt, fopt = pso(self._funcwrapped, self._lower, self._upper,
                maxiter=MAX_IT)


class PSOLSOptimizer(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'PSO-LS'

    def optimize(self):
        self._favor_context = True
        self._x, _= pso(self._funcwrapped, self._lower, self._upper,
                maxiter=MAX_IT, )
        self._favor_context = False
        self.lsearch()

    def lsearch(self):
        # Call here a local search to be fair in regards to the other
        # methods.
        res = optimize.minimize(fun=self._funcwrapped, x0=self._x,
                bounds=zip(self._lower, self._upper))

class PSOLSRestartOptimizer(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'PSO-LS-R'

    def optimize(self):
        self._favor_context = True
        while (self._nbcall < MAX_FN_CALL):
            x, v = pso(self._funcwrapped, self._lower, self._upper,
                maxiter=MAX_IT, )
            if self._xmini is None:
                self._xmini = x
            if self._fmini is None:
                self._fmini = v
            if v < self._fmini:
                self._fmini = v
                self._xmini = x

    def lsearch(self):
        # Call here a local search to be fair in regards to the other
        # methods.
        res = optimize.minimize(fun=self._funcwrapped, x0=self._xmini,
                bounds=zip(self._lower, self._upper))


class BHOptimizer(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'BH'

    def optimize(self):
        mybounds = MyBounds(self._lower, self._upper)
        res = optimize.basinhopping(self._funcwrapped, self._xinit,
            minimizer_kwargs={'method': 'L-BFGS-B',
                'bounds': [x for x in zip(self._lower, self._upper)]},
            accept_test=mybounds,
            niter = MAX_IT,
        )


class DEOptimizer(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'DE'

    def optimize(self):
        res = optimize.differential_evolution(self._funcwrapped,
            [x for x in zip(self._lower, self._upper)], maxiter=MAX_IT)


class DERestartOptimizer(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'DE-R'

    def optimize(self):
        while(self._nbcall < MAX_FN_CALL):
            res = optimize.differential_evolution(self._funcwrapped,
                [x for x in zip(self._lower, self._upper)],
                maxiter=MAX_IT)


class BFOptimizer(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'BF'

    def optimize(self):
        res = optimize.brute(self._funcwrapped,
            [x for x in zip(self._lower, self._upper)], )


def main():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)
    nbruns = 200
    bm = Benchmarker(nbruns=nbruns, folder='GENSA_bench_{}'.format(
       nbruns))
    bm.run()

if __name__ == '__main__':
    main()
