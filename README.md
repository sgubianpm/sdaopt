# PyGenSA

Simmulated Annealing Global optimization algorithm and extensive benchmark. Testing functions used in the benchmark have been implemented by Andreas Gavana, Andrew Nelson and scipy contributors and have been forked from SciPy project.
The generalized simulated annealing function is currently under proposed pull request to scipy:
https://github.com/scipy/scipy/pull/6569

Results of the benchmarks are available at:
https://gist.github.com/sgubianpm/7d55f8d3ba5c9de4e9f0f1ffff1aa6cf

Minimum requirements to run the benchmarks is to have scipy and matplotlib installed. Other dependencies are managed in the setup.py file. 
Running the benchmark is very CPU intensive and require a multicore machine or a cluster infrastructure.

## Installation

```bash
git clone https://github.com/sgubianpm/pygensa.git
cd pygensa
# Activate your appropriate python virtual environment if needed
python setup.py install
```

## Running benchmark on a multicore machine

```bash
# Activate your appropriate python virtual environment if needed
cd pygensa
# Replace NB_RUNS and OUTPUT_FODER by your values.
# NB_RUNS is the number of runs done for each testing function and algorithm used
# OUTPUT_FOLDER is the folder where result files will be generated
# If none of NB_RUNS and OUTPUT_FOLDER are provided, 50 runs and current working directory are used.
# The script uses all available cores on the machine.
python ./benchmark/workflow.py NB_RUNS OUTPUT_FOLDER
```

## Running benchmark on a cluster (Example for Moab/TORQUE)

The total number of testing functions is 261. The benchmark can be parallelized on 261 cores of the cluster infrastructure, the benchmarks are embarassingly parallel. If you cluster nodes have 16 cores, 17 sections will be required for splitting the processing (_261 / 16 = 16.3125, so 17 sections_)

Below a script content example for Maob/TORQUE:
```bash
#!/bin/bash
# Replace OUTPUT_FOLDER by your the path of your choice
# Adjust YOUR_PYTHON_VIRTUAL_ENV and YOUR_PYGENSA_GIT_FOLDER
##### These lines are for Moab
#MSUB -l procs=16
#MSUB -q long
#MSUB -o OUTPUT_FOLDER/bench.out
#MSUB -e OUTPUT_FOLDER/bench.err
source YOUR_PYTHON_VIRTUAL_ENV/bin/activate 
python YOUR_PYGENSA_GIT_FOLDER/benchmark/workflow.py 100 OUTPUT_FOLDER 
```
On your machine that is able to submit jobs to the cluster
```bash
for i in {0..16}
    do
        msub -v USE_CLUSTER,NB_CORES=16,SECTION_NUM=$i benchmark-cluster.sh
    done
```



