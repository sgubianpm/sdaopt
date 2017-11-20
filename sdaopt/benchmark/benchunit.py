# Copyright (c) 2017 Sylvain Gubian <sylvain.gubian@pmi.com>,
# Yang Xiang <yang.xiang@pmi.com>
# Author: Sylvain Gubian, PMP S.A.
# -*- coding: utf-8 -*-

import os
import logging
import pickle
import numpy as np

logger = logging.getLogger(__name__)

METRICS = [
        'success',
        'ncall',
        'fvalue',
        'time',
        'ncall_max',
        ]


class BenchUnit(object):
    def __init__(self, nbruns, name, algo, max_fn_call=1e6):
        self._name = name
        self._algo = algo
        self._nbruns = nbruns
        self._values = dict.fromkeys(METRICS)
        for k, v in self._values.items():
            self._values[k] = np.empty(nbruns)
            if k == 'success':
                self._values[k][:] = False
            elif k == 'ncall' or k == 'ncall_max':
                self._values[k][:] = max_fn_call
            elif k == 'fvalue':
                self._values[k][:] = 1e12
            else:
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
        # return int(np.amin(v))
        return np.round(np.amin(v), 6)

    @property
    def worst(self):
        v = self._values['ncall']
        v = v[np.where(self.success)]
        if not v.size:
            return np.inf
        # return int(np.amax(v))
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
    def medall(self):
        v = self._values['ncall']
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
