# Use-Case Examples

This directory contains examples of how to use the jaxDecomp library on a few use cases.

## Distributed LPT Cosmological Simulation

This example demonstrates the use of the 3D distributed FFT and halo exchange functions in the `jaxDecomp` library to implement a distributed LPT cosmological simulation. We provide a notebook to visualize the results of the simulation in [visualizer.ipynb](visualizer.ipynb).

To run the demo, some additional dependencies are required. You can install them by running:

```bash
pip install jax-cosmo
```

Then, you can run the example by executing the following command:
```bash
mpirun -n 4 python lpt_nbody_demo.py --nc 256 --box_size 256 --pdims 4x4 --halo_size 32 --output out
```

We also include an example of a slurm script in [submit_rusty.sbatch](submit_rusty.sbatch) that can be used to run the example on a slurm cluster with:
```bash
sbatch submit_rusty.sbatch
```
