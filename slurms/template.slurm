#!/bin/bash
#SBATCH --account=xyz@v100
#SBATCH --job-name=Learning-jax-SPMD     # Name of job
# Other partitions are usable by activating/uncommenting
# one of the 5 following directives:
#SBATCH -C v100-16g                 # decommenter pour reserver uniquement des GPU V100 16 Go
##SBATCH -C v100-32g                 # decommenter pour reserver uniquement des GPU V100 32 Go
##SBATCH --partition=gpu_p2          # decommenter pour la partition gpu_p2 (GPU V100 32 Go)
##SBATCH -C a100                  # decommenter pour reserver uniquement des GPU A100
# Ici, reservation de 8x10=80 CPU (4 taches par noeud) et de 8 GPU (4 GPU par noeud) sur 2 noeuds :
#SBATCH --nodes=2                  # nombre de noeud
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1       # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:4               # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
##SBATCH --gpus-per-task=2           # nombre de GPU par tache
# Le nombre de CPU par tache doit etre adapte en fonction de la partition utilisee. Sachant
# qu'ici on ne reserve qu'un seul GPU par tache (soit 1/4 ou 1/8 des GPU du noeud suivant
# la partition), l'ideal est de reserver 1/4 ou 1/8 des CPU du noeud pour chaque tache:
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (un quart du noeud ici)
##SBATCH --cpus-per-task=3           # nombre de CPU par tache pour gpu_p2 (1/8 du noeud 8-GPU)
##SBATCH --cpus-per-task=6           # nombre de CPU par tache pour gpu_p4 (1/8 du noeud 8-GPU)
##SBATCH --cpus-per-task=8           # nombre de CPU par tache pour gpu_p5 (1/8 du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=00:10:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=mpi_gpu_multi%j.out # name of output file
#SBATCH --error=mpi_gpu_multi%j.out  # name of error file (here, in common with the output file)
# Cleans out modules loaded in interactive and inherited by default
module purge

# Uncomment the following module command if you are using the "gpu_p5" partition
# to have access to the modules compatible with this partition.
#module load cpuarch/amd

# Loading modules
module load nvidia-compilers/23.9 cuda/11.8.0 cudnn/8.9.7.29-cuda  openmpi/4.1.1-cuda nccl/2.18.1-1-cuda cmake
module load python/3.10.4 && conda deactivate
source venv/bin/activate

# Echo of launched commands

set -x
# For the "gpu_p5" partition, the code must be compiled with the compatible modules.
# Code execution with binding via bind_gpu.sh : 1 GPU per task
srun python $1
