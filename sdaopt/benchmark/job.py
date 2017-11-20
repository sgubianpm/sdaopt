# Copyright (c) 2017 Sylvain Gubian <sylvain.gubian@pmi.com>,
# Yang Xiang <yang.xiang@pmi.com>
# Author: Sylvain Gubian, PMP S.A.
# -*- coding: utf-8 -*-

import logging
import multiprocessing


logger = logging.getLogger(__name__)


class Job(object):
    def __init__(self, name, klass, dim=None):
        self.name = name
        self.process = None
        self._status = 'INIT'
        self._klass = klass
        self.dim = dim

    def __str__(self):
        if self.dim is None:
            return 'JOB-{0}: {1}'.format(
                    self.index,
                    self.name,
                    )
        else:
            return 'JOB-{0}: {1} (DIM: {2})'.format(
                    self.index,
                    self.name, self.dim
                    )

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
