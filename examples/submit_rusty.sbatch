#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -t 0:10:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1

module load modules/2.3
module load gcc nvhpc python

source ~/venvs/jaxdecomp724/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LD_LIBRARY_PATH=$NVHPC_ROOT/Linux_x86_64/24.3/cuda/12.3/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/Linux_x86_64/24.3/cuda/12.3/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/Linux_x86_64/24.3/comm_libs/nccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/Linux_x86_64/24.3/math_libs/lib64:$LD_LIBRARY_PATH

mpirun python3 lpt_nbody_demo.py --pdims 2x2
