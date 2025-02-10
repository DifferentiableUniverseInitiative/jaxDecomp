
# Using the slurms

A quick guide on how to run jaxDecomp (this was only tested on Idris's Jean-Zay)

## Create a python environment

```bash
# if on A100
module load arch/a100
# then do this in this exact order
module load nvidia-compilers/23.9 cuda/12.2.0 cudnn/8.9.7.29-cuda  openmpi/4.1.5-cuda nccl/2.18.5-1-cuda cmake
# Then create your python env
module load python/3.10.4
# Make sure to use different envs for A100 and V100
python -m venv venv
conda deactivate
# Install dependencies
source venv/bin/activate
pip cache purge
# Installing jax
pip install jax[cuda]
# Then install jaxDecomp
pip install jaxdecomp
```

For an interactive use

on a100

```bash
salloc --account=xyz@a100 --nodes=1  --ntasks-per-node=4 --gres=gpu:4 -C a100 --hint=nomultithread --qos=qos_gpu-dev
```

on v100

```bash
salloc --account=xyz@v100 --nodes=1  --ntasks-per-node=4 --gres=gpu:4 -C v100-16g --hint=nomultithread --qos=qos_gpu-dev
```
or for a long dev session

```bash
salloc --account=xyz@v100 --time=04:00:00  --nodes=1  --ntasks-per-node=4 --gres=gpu:4 -C v100-32g --hint=nomultithread
```

## Once it is installed to load modules

Make sure to load the exact modules you used when you installed jaxDecomp

```bash
# if on A100
module load arch/a100
# then
module load nvidia-compilers/23.9 cuda/12.2.0 cudnn/8.9.7.29-cuda  openmpi/4.1.5-cuda nccl/2.18.5-1-cuda cmake
module load python/3.10.4 && conda deactivate
source venv/bin/activate
```

```bash
srun python -m pytest -s test_fft.py &> test_fft_${SLURM_PROCID}.log
```
