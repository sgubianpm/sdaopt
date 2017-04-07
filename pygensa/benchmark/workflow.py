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
from .bench import Benchmarker
from .benchstore import BenchStore
import subprocess
import argparse

__author__ = "Sylvain Gubian"
__copyright__ = "Copyright 2016, PMP SA"
__license__ = "GPL2.0"
__email__ = "Sylvain.Gubian@pmi.com"

logger = logging.getLogger(__name__)

DEFAULT_NB_RUNS = 100
DEFAULT_OUTPUT_FOLDER = os.path.join(os.getcwd(), 'DATA')
# If cluser is going to be used uncomment this section and
# launch the section number to be executed
# export env variables:
# export USE_CLUSTER=1
# export SECTION_NUM=0
# export NB_CORES=16

# Default settings will use the available cores on the local machine
# For high dimension benchmarking, few functions have been selected from
# the set where functions expression can be generalized for dimension n.

def run_all_bench():
    parser = argparse.ArgumentParser(
        description='Running benchmark and processing results')
    parser.add_argument(
        '--nb-runs',
        dest='nb_runs',
        action='store',
        type=int,
        default=DEFAULT_NB_RUNS,
        help='Number of runs for a given function to test by an algorithm'
    )
    parser.add_argument(
        '--output-folder',
        dest='output_folder',
        action='store',
        default=DEFAULT_OUTPUT_FOLDER,
        help='Folder where data file for optimization results are stored',
    )
    args = parser.parse_args()
    nb_runs = args.nb_runs
    output_folder = args.output_folder 

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)
    logger.warning(('The benchmark may take very long time depending on the'
                    ' number of cores available on your machine...'))
    
    bm = Benchmarker(nb_runs, output_folder)
    bm.run()
    # Generating the csv report file from the benchmark 
    path = os.path.join(output_folder, 'results.csv')
    logger.info('Reading all results data...')
    BenchStore.report(
        kind='csv', path=path, folder=output_folder)

    # Generate table and figure with Yang R script
    logger.info('Generating figure and table...')
    r_script_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        'scripts',
        'PyGenSA.R'))
    cmd = [
        'R', 'CMD', 'BATCH',
        r_script_path,
    ]
    logger.info('Command is: {0}'.format(' '.join(cmd)))
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    (out, err) = p.communicate()
    res = p.wait()
    if res !=0:
        print('Failed to run R script: {0}'.format(err))
    else:
        print('R script run successfuly')



# Call the main function with the first argument number of runs and second
# argument the folder path to results
if __name__ == '__main__':
    main(sys.argv)
