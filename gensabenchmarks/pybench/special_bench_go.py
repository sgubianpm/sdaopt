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
#import go_benchmark_functions as gbf

logger = logging.getLogger(__name__)

DEFAULT_TOL = 1.e-8
NB_CORES_AVAILABLES = multiprocessing.cpu_count() - 1
if NB_CORES_AVAILABLES < 1:
    NB_CORES_AVAILABLES = 1

METRICS = [
        'success',
        'ncall',
        'fvalue',
        'time',
        'ncall_max',
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

    def write(self, folder=os.getcwd()):
        filename = os.path.join(folder, '{0}-{1}-{2}.data'.format(
            self._name, self._algo, self._nbruns,))
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
                            bu.best, bu.mean, bu.worst, '{0} {1}'.format('(+/-)',
                            bu.std), bu.lowest, round(et,6)])
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
    def __init__(self, name, klass, ):
        self.name = name
        self.process = None
        self._status = 'INIT'
        self._klass = klass

    def __str__(self):
        return 'JOB-{0}: {1}'.format(self.index,
            self.name,)

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
                target=func, args=(self.name, self._klass,))
        logger.info('Starting job {0}'.format(self.name))
        self.process.start()
        self._status = 'STARTED'


class Benchmarker(object):
    def __init__(self, nbruns):
        self.algorithms = [BHOptimizer(), DEOptimizer(), GenSAOptimizer(),]
        #self.algorithms = [GenSAOptimizer(),]
        self.nbruns = nbruns
        bench_members = inspect.getmembers(gbf, inspect.isclass)
        self.benchmark_functions = [item for item in bench_members if
                                    issubclass(item[1], gbf.Benchmark)]

    def run(self):
        jobs = {}
        index = 0
        for name, klass in self.benchmark_functions:
            if name == 'Bukin06':
                continue
            jobs[index] = Job(name, klass)
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
            time.sleep(5)


    def bench(self, fname, klass):
        '''Benchmarking function. It executes all runs for a specific
        function
        '''
        self._fname = fname
        for algo in self.algorithms:
            bu = BenchUnit(self.nbruns, fname, algo.name)
            for i in range(self.nbruns):
                np.random.seed(1234 + i)
                algo.prepare(fname, klass)
                res = algo.optimize()
                if algo.fsuccess == np.inf:
                    logger.info(':-(  Func: {0} - Algo: {1} - RUN: {2} -> FAILED after {3} calls'.format(
                        fname, algo.name, i, algo.nbcall))
                    logger.info(res)
                else:
                    logger.info(':-)  Func: {0} - Algo: {1} - RUN: {2} -> FOUND after {3} calls'.format(
                        fname, algo.name, i, algo.nbcall))
                bu.update('success', i, algo.success)
                bu.update('ncall', i, algo.fcall_success)
                bu.update('fvalue', i, algo.fsuccess)
                bu.update('time', i, algo.duration)
                bu.update('ncall_max', i, algo.nbcall)
                bu.write('general_bench_{0}'.format(self.nbruns))

class OptimumFoundException(Exception):
    pass

class Algo(object):
    def __init__(self):
        self.name = None

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

    def prepare(self, fname, klass):
        self._k = klass()
        self._fname = fname
        self._nbcall = 0
        self._first_hit = True
        self._starttime = time.time()
        self._fcall_success = np.inf
        self._xsuccess = [np.inf] * self._k.N
        self._success = False
        self._fsuccess = np.inf
        self._hittime = np.inf
        self.recording = False
        self._values = []
        self._lower = -np.inf
        self._upper = np.inf

        self._lower = np.array([x[0] for x in self._k.bounds])
        self._upper = np.array([x[1] for x in self._k.bounds])
        self._xinit = self._lower + np.random.rand(self._k.N) * (
                self._upper - self._lower)

    def optimize(self):
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
        self.maxit = 5000
        Algo.optimize(self)
        try:
            ret = optimize.gensa(self._funcwrapped, None, self._lower,
                    self._upper, niter=self.maxit)
            return ret
        except OptimumFoundException:
            return None


class BHOptimizer(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'BasinHopping'

    def optimize(self):
        self.maxit = 5000
        mybounds = MyBounds(self._lower, self._upper)
        try:
            res = optimize.basinhopping(self._funcwrapped, self._xinit,
                minimizer_kwargs={'method': 'L-BFGS-B',
                    'bounds': [x for x in zip(self._lower, self._upper)]},
                #minimizer_kwargs={'method': 'BFGS'},
                accept_test=mybounds,
                niter = self.maxit,
            )
            return res
        except OptimumFoundException:
            return None

class DEOptimizer(Algo):
    def __init__(self):
        Algo.__init__(self)
        self.name = 'DifferentialEvolution'

    def optimize(self):
        self.maxit = 5000
        try:
            res = optimize.differential_evolution(self._funcwrapped,
                [x for x in zip(self._lower, self._upper)], maxiter=self.maxit)
            return res
        except OptimumFoundException:
            return None

def main():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)

    bm = Benchmarker(nbruns=200)
    bm.run()

if __name__ == '__main__':
    main()
