##############################################################################
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# Author: Sylvain Gubian, PMP SA
##############################################################################
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.md')).read()
CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()

requires = [
    'numpy',
    'scipy',
    'pytest',
    'pyswarm',
    'matplotlib',
    #  'fastcluster',
    ]

setup(
        name='pyhgsa',
        version='0.0.2',
        description='Hybrid General Simulated Annealing algorithm and benchmark',
        long_description=README + '\n\n' +  CHANGES,
        classifiers=[
            "Programming Language :: Python",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering :: Mathematics",
          ],
        author='Sylvain Gubian, PMP SA',
        author_email='sylvain.gubian@pmi.com',
        url='https://github.com/sgubianpm/pyhgsa',
        keywords='optimization benchmarking simulated annealing',
        packages=find_packages(),
        include_package_data=True,
        entry_points = {
            'console_scripts':
            ['hgsa_bench=pyhgsa.benchmark.workflow:run_all_bench'],
        },
        zip_safe=False,
        install_requires=requires,
        tests_require=requires,
        test_suite="tests",
      )

