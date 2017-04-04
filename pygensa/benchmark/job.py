##############################################################################
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
##############################################################################
# -*- coding: utf-8 -*-
import logging
import multiprocessing

__author__ = "Sylvain Gubian"
__copyright__ = "Copyright 2016, PMP SA"
__license__ = "GPL2.0"
__email__ = "Sylvain.Gubian@pmi.com"

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
