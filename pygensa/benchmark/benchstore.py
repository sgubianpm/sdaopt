##############################################################################
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
##############################################################################
# -*- coding: utf-8 -*-
import logging
import os
import pickle
import csv
import numpy as np

__author__ = "Sylvain Gubian"
__copyright__ = "Copyright 2016, PMP SA"
__license__ = "GPL2.0"
__email__ = "Sylvain.Gubian@pmi.com"

logger = logging.getLogger(__name__)


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
            with open(path, 'w') as csvfile:
                csvwriter = csv.writer(
                        csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL
                        )
                csvwriter.writerow([
                    'Function name', 'Algorithm', 'Success Rate', 'Best',
                    'Average', 'Worst', 'Std', 'fvalue (mean)',
                    'Mean time (ms)', 'Median All',
                    ])
                for f in files:
                    with open(os.path.join(folder, f), 'rb') as fh:
                        bu = pickle.load(fh)
                        logger.info('Processing benchunit: {0}'.format(bu))
                        et = bu.time
                        if et < 0:
                            et = np.inf
                        else:
                            et *= 1000
                        success_rate = np.round(np.where(
                            bu.success)[0].size * 100 / bu.success.size, 1)
                        csvwriter.writerow([
                            bu.name, bu.algo,
                            "{0}%".format(success_rate),
                            bu.best, bu.mean, bu.worst,
                            '{0} {1}'.format('(+/-)', bu.std), bu.lowest,
                            round(et, 6), bu.medall
                            ])
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
                    success_rate = np.round(
                            np.where(bu.success)[0].size *
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
                    table.append([
                        name, algo,
                        "{0}%".format(success_rate),
                        "{0}".format(bu.best),
                        "{0}".format(bu.med),
                        "{0}".format(bu.worst), '{0} {1:.2f}'.format(
                            '+/-',
                            bu.std),
                        ])
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
