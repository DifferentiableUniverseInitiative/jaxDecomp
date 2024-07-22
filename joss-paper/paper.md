---
title: 'jaxDecomp : JAX Library for 3D Domain Decomposition and Parallel FFTs'
tags:
  - Jax
  - CUDA
  - Python
  - HPC
  - FFT
  - Simulations
authors:
  - name: Wassim Kabalan
    orcid: 0009-0001-6501-4564
    affiliation: 1
  - name: François Lanusse
    orcid: 0000-0001-7956-0542
    affiliation: 2, 3
  - name: Alexandre Boucaud
    affiliation: 1
  - name: Eric Aubourg
    affiliation: 1
affiliations:
 - name: Université Paris Cité, CNRS, Astroparticule et Cosmologie, F-75013 Paris, France
   index: 1
 - name: Université Paris-Saclay, Université Paris Cité, CEA, CNRS, AIM, 91191, Gif-sur-Yvette, France
   index: 2
 - name: Flatiron Institute, Center for Computational Astrophysics, 162 5th Avenue, New York, NY 10010, USA
   index: 3
date: 26 June 2024
bibliography: paper.bib

---


# Abstract


`JAX` [@JAX] has seen widespread adoption in both machine learning and scientific computing due to its flexibility and performance, as demonstrated in projects like `JAX-Cosmo` [@JAXCOSMO]. However, its application in distributed high-performance computing (HPC) has been limited by the complex nature of inter-GPU communications required in HPC scientific software, which is more challenging compared to deep learning networks. Previous solutions, such as `MPI4JAX` [@mpi4jax], provided support for single program multiple data (SPMD) operations but faced significant scaling limitations.


Recently, JAX has made a major push towards simplified SPMD programming, with the unification of the JAX array API and the introduction of several powerful APIs, such as `pjit`, `shard_map`, and `custom_partitioning`. However, not all native JAX operations have specialized distribution strategies, and `pjitting` a program can lead to excessive communication overhead for some operations, particularly the 3D Fast Fourier Transform (FFT), which is one of the most critical and widely used algorithms in scientific computing. Distributed FFTs are essential for many simulation and solvers, especially in fields like cosmology and fluid dynamics, where large-scale data processing is required.

To address these limitations, we introduce jaxDecomp, a JAX library that wraps NVIDIA's `cuDecomp` domain decomposition library [@cuDecomp]. jaxDecomp provides JAX primitives with highly efficient CUDA implementations for key operations such as 3D FFTs and halo exchanges. By integrating seamlessly with JAX, jaxDecomp supports running on multiple GPUs and nodes, enabling large-scale, distributed scientific computations. Implemented as JAX primitives, jaxDecomp builds directly on top of the distributed Array strategy in JAX and is compatible with JAX transformations such as `jax.grad` and `jax.jit`, ensuring fast execution and differentiability with a python, easy-to-use interface. Using cuDecomp, jaxDecomp can switch between `NCCL`, `CUDA-Aware MPI` and `NVSHMEM` for distributed array transpose operations, allowing it to best fit the specific HPC cluster configuration.

# Statement of Need

For numerical simulations on HPC systems, having a distributed, easy-to-use, and differentiable FFT is critical for achieving peak performance and scalability. While it is technically feasible to implement distributed FFTs using native JAX, for performance and memory-critical simulations, it is better to use specialized HPC codes. These codes, however, are not typically differentiable. The need for differentiable, performant, and memory-efficient code has risen due to the recent introduction of differentiable algorithms such as Hamiltonian Monte Carlo (HMC) [@HMC] and the No-U-Turn Sampler (NUTS) [@NUTS].

In scientific applications such as particle mesh (PM) simulations for cosmology, existing frameworks like `FlowPM` [@FlowPM], a `TensorFlow-mesh` [@TF-MESH] based simulation, are distributed but no longer actively maintained. Similarly, JAX-based frameworks like `pmwd` [@pmwd] are limited to 512 volumes due to the lack of distribution capabilities. These examples underscore the critical need for scalable and efficient solutions. jaxDecomp addresses this gap by enabling distributed and differentiable 3D FFTs within JAX, thereby facilitating the simulation of large cosmological volumes on HPC clusters effectively.

While it is technically feasible to implement distributed FFTs using native JAX, there are significant benefits to using jaxDecomp. Although the performance difference may be marginal, jaxDecomp offers several advantages that make it a valuable tool for HPC applications. Firstly, jaxDecomp provides the ability to easily switch backends between NCCL, MPI, and NVSHMEM, optimizing performance based on the specific HPC cluster configuration. Secondly, jaxDecomp performs operations in place, which is more memory-efficient, minimizing the use of intermediate memory and enhancing overall performance. This is crucial for memory-bound codes such as cosmological simulations.

# Implementation

## Distributed FFT Algorithm

The following steps outline the distributed FFT algorithm in jaxDecomp, which uses 2D domain decomposition to distribute 3D data across GPUs.

| Steps            | Local Operation                                 | Global Operation                                                               |
|------------------|-------------------------------------------------|--------------------------------------------------------------------------------|
| FFT along X      | Perform batched FFT along the X axis.           | -                                                                              |
| Transpose X to Y | Local transpose to $Y \times X \times Z$        | All-to-all communication to concatenate $Y$: $Y \times \frac{X}{P_y} \times \frac{Z}{P_z}$ |
| FFT along Y      | Perform batched FFT along the Y axis.           | -                                                                              |
| Transpose Y to Z | Local transpose to $Z \times X \times Y$        | All-to-all communication to concatenate $Z$: $Z \times \frac{X}{P_z} \times \frac{Y}{P_y}$ |
| FFT along Z      | Perform batched FFT along the Z axis.           | -                                                                              |




Each transpose includes a local cyclic transposition of axes, which implies a transposition of the decomposition grid. This process involves both a local transposition and a processor grid transposition at each step.

In order to capture the changes and return the right output to JAX in a transparent way, the local transpositions are described in the lowering of the primitive and the GPU grid transpositions are described in the `infer_sharding_from_operands` rule, which is part of JAX's `custom_partitioning` API.

![Visualization of the distributed FFT process in jaxDecomp](assets/fft.svg)

## Distributed Halo Exchange

The halo exchange is a crucial step in distributed programming. It allows the transfer of data on the edges of each slice to the adjacent slice, ensuring data consistency across the boundaries of distributed domains.

Many applications in high-performance computing (HPC) use domain decomposition to distribute the workload among different processing elements. These applications, such as cosmological simulations, stencil computations, and PDE solvers, require the halo regions to be updated with data from neighboring regions. This process, often referred to as a halo update, is implemented using MPI (Message Passing Interface) on large machines.

Using cuDecomp, we can also change the communication backend to `NCCL`, `MPI`, or `NVSHMEM`.

### Halo Exchange Process

For each axis, a slice of data of size equal to the halo extent is exchanged between neighboring subdomains.

![Visualization of the distributed halo exchange process in jaxDecomp](assets/halo-exchange.svg)

### Efficient State Management in jaxDecomp

jaxDecomp effectively manages the metadata and resources required for cuDecomp operations, ensuring both efficiency and performance. This is achieved through a caching mechanism that stores the necessary information for transpositions and halo exchanges, as well as cuFFT plans.

jaxDecomp caches the metadata that cuDecomp uses for transpositions and halo exchanges, and also caches the cuFFT plans. All this data is created efficiently and lazily (i.e., it is generated only when needed during JAX's just-in-time (JIT) compilation of functions) and stored for subsequent use. This approach ensures that resources are allocated only when necessary, reducing overhead and improving performance.

The cached data is properly destroyed at the end of the session, ensuring that no resources are wasted or leaked.

Additionally, jaxDecomp opportunistically creates inverse FFT (IFFT) plans when the FFT is JIT compiled. This leads to improved performance, as the IFFT plans are readily available for use, resulting in a 5x speedup in the IFFT JIT compilation process.


# API description

In this description, we show how to perform a distributed 3D FFT using `jaxDecomp` and `JAX`. The code snippet below demonstrates the initialization of the distributed mesh, the creation of the initial distributed tensor, and the execution of a distributed 3D FFT using `jaxDecomp`.

```python
import jax
import jaxdecomp

# Setup
master_key = jax.random.PRNGKey(42)
key = jax.random.split(master_key, size)[rank]
pdims = (2 , 2)
mesh_shape = [2048, 2048 , 2048]
halo_size = (256 , 256 , 0)

# Create computing mesgh
devices = mesh_utils.create_device_mesh(pdims)
mesh = Mesh(devices, axis_names=('y', 'z'))
sharding = jax.sharding.NamedSharding(mesh, P('z', 'y'))

### Create all initial distributed tensors ###
local_mesh_shape = [
    mesh_shape[0] // pdims[1], mesh_shape[1] // pdims[0], mesh_shape[2]
]

z = jax.make_array_from_single_device_arrays(
        shape=mesh_shape,
        sharding=sharding,
        arrays=[jax.random.normal(key, local_mesh_shape)])


@jax.jit
def do_fft(z):
      k_array = jaxdecomp.fft.pfft3d(z)
      k_array = k_array * 2 # element wise operation is distributed automatically by jax
      r_array = jaxdecomp.fft.pifft3d(k_array).real
      return r_array


def do_halo_exchange(z):
    padding = ((halo_size[0] , halo_size[0]) , (halo_size[1] , halo_size[1]) , (0 , 0))
    z = jaxdecomp.slice_pad(k_array , padding , pdims)
    z = jaxdecomp.halo_exchange(
                        z,
                        halo_extents=(halo_size // 2 , halo_size // 2),
                        halo_periods=(True, True, True),
                        reduce_halo=True)
    z = jaxdecomp.slice_unpad(z , padding , pdims)
    return k_array

def do_transpose(x_pencil):
    y_pencil = jaxdecomp.transposeXtoY(x_pencil)
    z_pencil = jaxdecomp.transposeYtoZ(y_pencil)
    y_pencil = jaxdecomp.transposeZtoY(z_pencil)
    x_pencil = jaxdecomp.transposeYtoX(y_pencil)
    return x_pencil



with mesh:
    z = do_fft(z)
    z = do_halo_exchange(z)
    z = do_transpose(z)

```

# Example

In the provided example, the code computes the gravitational potential using Fast Fourier Transforms (FFT) within a JAX-based environment. The code can run on multiple GPUs and nodes using `jaxDecomp` and `JAX` while being fully differentiable.


```python
def potential(delta):
  delta_k = pfft3d(delta)
  ky, kz, kx= [jnp.fft.fftfreq(s) * 2 * np.pi for s in delta.shape]
  laplace_kernel = jnp.where(kk == 0, 1., 1. / -(kx**2 + ky**2 + kz**2))
  potential_k = delta_k * laplace_kernel
  return ipfft3d(potential_k)
```


A more detailed example of a LPT simulation can be found in the [jaxdecomp_lpt](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/blob/main/examples/lpt_nbody_demo.py).


### Benchmarks

The performance benchmarks for jaxDecomp were conducted on the Jean Zay supercomputer using the A100 GPUs (each with 80 GB of memory). These tests demonstrate the scalability and efficiency of jaxDecomp in handling large-scale FFT operations.

The tests show that jaxDecomp scales very well, even when distributed across multiple nodes, maintaining efficient performance. In particular, jaxDecomp demonstrates efficient computation as the number of GPUs and problem size increase.

![Strong Scaling Performance of jaxDecomp](assets/strong_scaling.png)



![Weak Scaling Performance of jaxDecomp](assets/weak_scaling.png)


# Stability and releases

A lot of effort has been put into packaging and testing. We aim to have a 100% code coverage tests covering all four functionalities (FFT, Halo, (un)padding, and transposition). The code has been tester on the Jean Zay supercomputer, and we have been able to run simulations up to 64 GPUs.
We also aim to package the code and release it on PyPI as built wheels for HPC clusters.

# Acknowledgements

This work was granted access to the HPC resources of IDRIS under the allocation 2024-AD011014949 made by GENCI. The computations in this work were, in part, run at facilities supported by the Scientific Computing Core at the Flatiron Institute, a division of the Simons Foundation.

We also acknowledge the SCIPOL scipol.in2p3.fr funded by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (PI: Josquin Errard, Grant agreement No. 101044073).

# References
