##############################################################################
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
##############################################################################
# -*- coding: utf-8 -*-
import os
import sys
import logging
from benchmark.bench import Benchmarker
from benchmark.benchstore import BenchStore

__author__ = "Sylvain Gubian"
__copyright__ = "Copyright 2016, PMP SA"
__license__ = "GPL2.0"
__email__ = "Sylvain.Gubian@pmi.com"

logger = logging.getLogger(__name__)

NB_RUNS = 2
OUTPUT_FOLDER = os.getcwd()
# If cluser is going to be used uncomment this section and
# launch the section number to be executed
# export env varaibles:
# export USE_CLUSTER=1
# export SECTION_NUM=0
# export NB_CORES=16

# Default settings will use the available cores on the local machine
# For high dimension benchmarking, few functions have been selected from
# the set where functions expression can be generalized for dimension n.

def main():
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)
    
    bm = Benchmarker(NB_RUNS, OUTPUT_FOLDER)
# This may take a long time depending of NB_RUNS and number of cores available
    bm.run()
    # Generating the csv report file from the benchmark 
    path = os.path.join(OUTPUT_FOLDER, 'results.csv')
    BenchStore.report(
        kind='csv', path=path, folder=OUTPUT_FOLDER)

if __name__ == '__main__':
    main()
