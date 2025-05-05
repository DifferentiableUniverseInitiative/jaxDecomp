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


Here is the updated version of the **Abstract**, incorporating the advisor’s handwritten corrections and suggestions from the image, as well as the formatting requirement to enclose `jaxDecomp` and `cuDecomp` in backticks (` `):

---

# Abstract


`JAX` \[@JAX] has seen widespread adoption in both machine learning and scientific computing due to its flexibility and performance, as demonstrated in projects like `JAX-Cosmo` \[@JAXCOSMO]. However, its application in distributed high-performance computing (HPC) has been limited by the complex nature of inter-GPU communications required in HPC scientific software, which is more challenging compared to that of deep learning networks. Proposed solutions such as `MPI4JAX` \[@mpi4jax] and `cuFFTMP` have enabled distributed computing capabilities, with `cuFFTMP` offering high-performance 3D FFTs via NVSHMEM and `MPI4JAX` supporting general SPMD operations using MPI. Nonetheless, both approaches face significant limitations: `MPI4JAX` is constrained by the MPI protocol’s handling of large messages—typically capped at 2–4 GB per buffer on many systems—while `cuFFTMP` requires explicit NVSHMEM-based memory management, making integration with the JAX ecosystem more difficult.

With the introduction of the unified JAX array API in JAX v0.4, along with powerful tools such as `pjit`, `shard_map`, and `custom_partitioning`, SPMD programming has become more accessible. Nonetheless, many NumPy \[@NUMPY] and SciPy \[@SCIPY] operators used in scientific computing are often applied to distributed arrays but will internally gather the entire array onto a single device, defeating the purpose of parallelizing large computations. This feature is particularly problematic for the 3D Fast Fourier Transform (FFT)—a critical operation in scientific computing. FFTs are a core component of large-scale simulations in fields like cosmology and fluid dynamics \[@jax-md], where the data arrays typically exceed the memory capacity of a single device—making any implicit gathering onto one device both inefficient and fundamentally incompatible with the goals of distributed FFT computation.

To address these challenges, we present `jaxDecomp`, a JAX library that wraps NVIDIA’s `cuDecomp` domain decomposition library \[@cuDecomp] and implements all operators as JAX primitives. `jaxDecomp` integrates specialized HPC code into the JAX ecosystem for key distributed operations, including 3D FFTs and halo exchanges. It scales seamlessly to multiple GPUs and nodes. By building on the distributed array strategy in JAX, it remains compatible with standard JAX transformations such as `jax.grad` and `jax.jit`, ensuring a Pythonic, differentiable interface. `jaxDecomp` supports NCCL, CUDA-Aware MPI, or NVSHMEM for inter-GPU data transposition, providing flexibility to match a range of HPC cluster configurations.

In our benchmarks, `jaxDecomp` demonstrates strong performance while being trivial to install and use. It fills a critical gap by offering a lightweight and fully differentiable library for distributed 3D FFTs, halo exchanges, and related operations in JAX. This seamless approach simplifies large-scale simulation workflows, optimizes performance across diverse hardware setups, and eliminates the need for manual distributed FFT implementations—while preserving memory efficiency and ease of use.



# Statement of Need

For numerical simulations on HPC systems, a distributed, easy-to-use, and differentiable FFT is essential for achieving peak performance and scalability. There is a pressing need for a solution that can serve as a true drop-in replacement for `jax.numpy.fft` and install seamlessly—especially for HPC users who must integrate efficiently with existing cluster infrastructure.

In scientific applications such as cosmological particle mesh (PM) simulations, specialized frameworks like `FlowPM` [@FlowPM] built on `mesh-TensorFlow` [@TF-MESH] or JAX-based codes like `pmwd` [@pmwd] often struggle to scale beyond single-node memory limits or rely on manual distribution strategies. These challenges highlight the need for a scalable, high-performance approach to distributed FFTs that remains differentiable for advanced algorithms (like Hamiltonian Monte Carlo [@HMC] or the No-U-Turn Sampler (NUTS) [@NUTS]).

`jaxDecomp` fills this gap by offering a lightweight JAX library that wraps NVIDIA’s `cuDecomp` for distributed 3D FFTs, halo exchanges, and related operations, all implemented as fully differentiable JAX primitives. This seamless approach lets users transparently switch between NCCL, MPI, or NVSHMEM backends, optimizing performance for the specific setup of each HPC cluster. As a result, `jaxDecomp` simplifies large-scale simulation workflows and eliminates the need to manually implement distributed FFTs in JAX, while preserving memory efficiency, compatibility with JAX transformations, and ease of use.


# Implementation

## Distributed FFT Algorithm

The distributed FFT in `jaxDecomp` is performed by applying a series of 1D FFTs along each of the three axes of a 3D array: first along the Z-axis, then the Y-axis, and finally the X-axis. Local 1D FFTs are performed along the axis that is currently undistributed, while global transpositions are used to realign the data so that each axis becomes undistributed in turn. This sequence ensures that each FFT operates locally, with global transpositions enabling the redistribution of data between steps.

This sequence is illustrated in the following figure, showing both the forward and backward passes of the distributed 3D FFT:

![Visualization of the 3D FFT algorithm in `jaxDecomp`, including forward and backward passes via axis-aligned transpositions and local 1D FFTs.](assets/fft.svg)


### Data Transposition

To effectively implement the distributed FFT, the data must be transposed between each FFT operation, ensuring the correct alignment of the distributed and undistributed axes. These transpositions involve both local cyclic adjustments on each GPU and global communications across the processor grid.

The following table outlines the transposition steps involved in `jaxDecomp`, which rearranges the data to facilitate the distributed FFT process:


| Steps            | Operation Description                                    |
|------------------|----------------------------------------------------------|
| FFT along Z      | Batched 1D FFT along the Z-axis.                         |
| Transpose Z to Y | Transpose to $Z \times X \times Y$. Partition the Y-axis |
| FFT along Y      | Batched 1D FFT along the Y-axis.                         |
| Transpose Y to X | Transpose to $Y \times Z \times X$. Partition the X-axis |
| FFT along X      | Batched 1D FFT along the X-axis.                         |


### Domain Decomposition

Domain decomposition is a method used in parallel computing to break down large computational domains into smaller subdomains, facilitating efficient data distribution across multiple GPUs. In the context of 3D FFTs, domain decomposition and transpositions work together to manage the data, with transpositions gathering one axis while distributing another for the FFT.

In `jaxDecomp`, the Z-axis always starts as undistributed, and the decomposition dimensions are defined by the $P_x$ and $P_y$ parameters.

#### Pencil Decomposition

In pencil decomposition, the 3D FFT is computed via three sequential 1D FFTs, each separated by a transposition that redistributes the data to align the next undistributed axis. The 1D FFT is performed on the fastest (inner-most) axis, which is undistributed at that stage of the algorithm (e.g., X for the X-pencil, Y for the Y-pencil, etc.).

| Step             | Origin     | Target      |
|----------------------|-----------------------------------------------|------------------------------------------|
| Transpose Z to Y | $\frac{X}{P_x} \times \frac{Y}{P_y} \times Z$| $\frac{Z}{P_y} \times \frac{X}{P_x} \times Y$ |
| Transpose Y to X | $\frac{Z}{P_y} \times \frac{X}{P_x} \times Y$| $\frac{Y}{P_x} \times \frac{Z}{P_y} \times X$ |
| Transpose X to Y | $\frac{Y}{P_x} \times \frac{Z}{P_y} \times X$| $\frac{Z}{P_y} \times \frac{X}{P_x} \times Y$ |
| Transpose Y to Z | $\frac{Z}{P_y} \times \frac{X}{P_x} \times Y$| $\frac{X}{P_x} \times \frac{Y}{P_y} \times Z$ |

#### Slab Decomposition

For 1D decomposition (slabs), we need to perform one 1D FFT and one 2D FFT. 2D FFTs present additional challenges because both the fastest and second-fastest axes must remain undistributed.

For example, consider a $(P_y, P_z)$ decomposition with $P_z = 1$:

| Step            | Decomposition                     | FFT Feasibility                 |
|-----------------|-----------------------------------|---------------------------------|
| Initial         | $X \times \frac{Y}{P_y} \times Z$  | only 1D FFT on Z        |
| Transpose Z to Y| $\frac{Z}{P_y} \times X \times Y$  | 2D FFT on YX            |

This is the case for the YZ slab, where the transformation sequence enables the application of a 2D FFT on the YX plane:

2D FFT($YX$) $\rightarrow$ TransposeYtoX $\rightarrow$ 1D FFT($X$)

For the other decomposition, with $P_y = 1$ and $P_x = n$ where $n$ is the number of GPUs :

| Step            | Decomposition                    | FFT Feasibility                    |
|-----------------|----------------------------------|------------------------------------|
| Initial         | $\frac{X}{P_x} \times Y \times Z$| 1D FFT on Z or 2D FFT on YX |
| Transpose Z to Y| $Z \times \frac{X}{P_x} \times Y$| 1D FFT on Y                 |

Since `cuDecomp` does not support a direct transposition from a Z pencil to an X pencil, we achieve this by applying a coordinate transformation that effectively reinterprets the axes from XYZ to YZX—enabling slab decomposition in a single transposition step.


#### Slab Decomposition with Coordinate Transformation

| Step               | Decomposition                        | Our Coordinates                          | Coordinate Step      | FFT Feasibility                        |
|--------------------|--------------------------------------|------------------------------------------|----------------------|----------------------------------------|
| Initial            | $\frac{Z}{P_x} \times X \times Y$     | $\frac{X}{P_x} \times Y \times Z$       | –                    | 2D FFT on ZY                    |
| Transpose Y to Z   | $X \times \frac{Y}{P_x} \times Z$     | $Y \times \frac{Z}{P_x} \times X$       | Transpose Z to X     | 1D (I)FFT on the last axis X     |
| Transpose Z to Y   | $\frac{Z}{P_x} \times X \times Y$     | $\frac{X}{P_x} \times Y \times Z$       | Transpose X to Z     | 2D IFFT on ZY                   |

This approach ensures that slab decomposition can be achieved in a single transposition step, enhancing computational efficiency.


2D FFT($ZY$) $\rightarrow$ TransposeZtoX $\rightarrow$ 1D FFT($X$)

### Non-Contiguous Global Transpose

`jaxDecomp` also supports non-contiguous transpositions, where the transposition is performed globally across devices, without requiring intermediate local reshuffling on each GPU. In this case, the $P_x$ and $P_y$ dimensions remain associated with their original axes throughout the process, maintaining the same axis order (`X`, `Y`, `Z`). This method is particularly useful in workflows that benefit from preserving the global logical layout of the array, such as halo exchanges.

The following table illustrates the steps for a non-contiguous global transpose, where the $P_x$ and $P_y$ dimensions stay aligned with the `X`, `Y`, and `Z` axes, without any permutation:

| Step       | Origin                                       | Target                                         |
|------------|----------------------------------------------|------------------------------------------------|
| Transpose Z to Y     | $\frac{X}{P_x} \times \frac{Y}{P_y} \times Z$ | $\frac{X}{P_x} \times Y \times \frac{Z}{P_y}$   |
| Transpose Y to X     | $\frac{X}{P_x} \times Y \times \frac{Z}{P_y}$ | $X \times \frac{Y}{P_x} \times \frac{Z}{P_y}$   |
| Transpose X to Y     | $X \times \frac{Y}{P_x} \times \frac{Z}{P_y}$ | $\frac{X}{P_x} \times Y \times \frac{Z}{P_y}$   |
| Transpose Y to Z     | $\frac{X}{P_x} \times Y \times \frac{Z}{P_y}$ | $\frac{X}{P_x} \times \frac{Y}{P_y} \times Z$   |

Our benchmarks did not show any significant performance difference between the contiguous and non-contiguous transpositions. As a result, non-contiguous transposes can simplify implementation without compromising performance.


## Distributed Halo Exchange

The halo exchange is a crucial step in distributed programming. It allows the transfer of data on the edges of each slice to the adjacent slice, ensuring data consistency across the boundaries of distributed domains.

Many applications in high-performance computing (HPC) use domain decomposition to distribute the workload among different processing elements. These applications, such as cosmological simulations, stencil computations, and PDE solvers, require the halo regions to be updated with data from neighboring regions. This process, often referred to as a halo update, is implemented using MPI (Message Passing Interface) on large machines.

Using `cuDecomp`, we can flexibly switch the communication backend between `NCCL`, `MPI`, or `NVSHMEM`, depending on the target hardware and cluster configuration. Some HPC systems are highly optimized for `NCCL`—particularly those with NVIDIA GPUs and fast NVLink interconnects—while others are designed around traditional `MPI`-based infrastructure. `NVSHMEM`, on the other hand, enables one-sided communication and memory sharing between GPUs, which can reduce synchronization overhead and improve performance in certain stencil or halo-exchange-heavy workloads.


### Halo Exchange Process

For each axis, `jaxDecomp` performs a bidirectional halo exchange, where a slice of width equal to the halo extent is exchanged with adjacent subdomains. This ensures that each subdomain has access to the necessary boundary data from its neighbors.

The following table shows the index ranges involved in each send and receive operation:

| Direction            | Sent Range (from current slice)                | Received Range (into current slice)                |
|----------------------|------------------------------------------------|----------------------------------------------------|
| To next neighbor     | $[S - 2 \cdot h : S - h]$  | $[0 : h]$ (from previous neighbor)                 |
| To previous neighbor | $[h : 2 \cdot h]$                               | $[S - h : S]$ (from next neighbor) |

Where :

 - $h$ is the **halo extent**
 - $S$ is the local size of the array along the axis

![Visualization of the distributed halo exchange process in `jaxDecomp`](assets/halo-exchange.svg)

### Efficient State Management

jaxDecomp effectively manages the metadata and resources required for `cuDecomp` operations, ensuring both efficiency and performance. This is achieved through a caching mechanism that stores the necessary information for transpositions and halo exchanges, as well as cuFFT plans.

jaxDecomp caches the metadata that `cuDecomp` uses for transpositions and halo exchanges, and also caches the cuFFT plans. All this data is created efficiently and lazily (i.e., it is generated only when needed during JAX's just-in-time (JIT) compilation of functions) and stored for subsequent use. This approach ensures that resources are allocated only when necessary, reducing overhead and improving performance.

The cached data is properly destroyed at the end of the session, ensuring that no resources are wasted or leaked.

Additionally, `jaxDecomp` opportunistically creates inverse FFT (IFFT) plans when the FFT is JIT compiled. This leads to improved performance, as the IFFT plans are readily available for use, resulting in a 5x speedup in the IFFT JIT compilation process.


### Benchmarks

The performance benchmarks for `jaxDecomp` were conducted on the Jean Zay supercomputer using NVIDIA A100 GPUs (each with 80 GB of memory). These tests evaluated both strong and weak scaling of large-scale 3D FFT operations across multiple nodes.

We benchmarked both backends available in `jaxDecomp`: the `cuDecomp`-based implementation and a pure JAX-based backend. The benchmarks indicate that `cuDecomp` is slightly faster than native JAX, particularly for large, multi-node workloads.
![*Strong scaling results on the Jean Zay supercomputer using A100 GPUs.*](assets/strong_scaling.png){ width=100% }

![*Weak scaling results showing that `jaxDecomp` maintains high efficiency as both problem size and GPU count increase.*](assets/weak_scaling.png){ width=100% }



# Stability and releases

A lot of effort has been put into packaging and testing. We aim to have a 100% code coverage tests covering all four functionalities (FFT, Halo, (un)padding, and transposition). The code has been tester on the Jean Zay supercomputer, and we have been able to run simulations up to 64 GPUs.
We also aim to package the code and release it on PyPI as built wheels for HPC clusters.

# Acknowledgements

This work was granted access to the HPC resources of IDRIS under the allocation 2024-AD011014949 made by GENCI. The computations in this work were, in part, run at facilities supported by the Scientific Computing Core at the Flatiron Institute, a division of the Simons Foundation.

We also acknowledge the SCIPOL scipol.in2p3.fr funded by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (PI: Josquin Errard, Grant agreement No. 101044073).


### Appendix

Detailed examples demonstrating the usage of `jaxDecomp` are provided in the appendix:

- **Appendix A**: Performing a distributed 3D FFT with `jaxDecomp`
- **Appendix B**: Particle-Mesh (PM) simulation example using distributed FFTs



## Appendix A: API description

In this description, we show how to perform a distributed 3D FFT using `jaxDecomp` and `JAX`. The code snippet below demonstrates the initialization of the distributed mesh, the creation of the initial distributed tensor, and the execution of a distributed 3D FFT using `jaxDecomp`.

```python
import jax

jax.distributed.initialize()
import jaxdecomp

P = jax.sharding.PartitionSpec

rank = jax.process_index()
size = jax.process_count()
# Setup
master_key = jax.random.PRNGKey(42)
key = jax.random.split(master_key, size)[rank]
pdims = (2, 2)
mesh_shape = [2048, 2048, 2048]
halo_size = (128, 128)

# Create computing mesh
mesh = jax.make_mesh(pdims, ('y', 'z'))
sharding = jax.sharding.NamedSharding(mesh, P('z', 'y'))

### Initialize distributed tensors
local_mesh_shape = [mesh_shape[0] // pdims[1], mesh_shape[1] // pdims[0], mesh_shape[2]]

z = jax.make_array_from_single_device_arrays(shape=mesh_shape, sharding=sharding, arrays=[jax.random.normal(key, local_mesh_shape)])


@jax.jit
def do_fft(z):
    k_array = jaxdecomp.fft.pfft3d(z)
    # element wise operation is distributed automatically by jax
    k_array = k_array * 2
    r_array = jaxdecomp.fft.pifft3d(k_array).real
    return r_array


def do_halo_exchange(z):
    z = jaxdecomp.halo_exchange(z, halo_extents=halo_size, halo_periods=(True, True))
    return z


def do_transpose(x_pencil):
    y_pencil = jaxdecomp.transposeXtoY(x_pencil)
    z_pencil = jaxdecomp.transposeYtoZ(y_pencil)
    y_pencil = jaxdecomp.transposeZtoY(z_pencil)
    x_pencil = jaxdecomp.transposeYtoX(y_pencil)
    return x_pencil


# Perform a distributed 3D FFT, scale in Fourier space, and inverse FFT
z = do_fft(z)
# Perform halo exchange to update boundary values with neighbors
z = do_halo_exchange(z)
# Apply a round-trip of distributed transpositions (X → Y → Z → Y → X)
z = do_transpose(z)

```

# Appendix B: Particle-Mesh Example (PM Forces)

In the following example, the code computes gravitational forces using a Particle-Mesh (PM) scheme within a JAX-based environment. The code can run on multiple GPUs and nodes using `jaxDecomp` and `JAX` while remaining fully differentiable.

This method is particularly relevant for particle-mesh simulations, as demonstrated in the PMWD paper. The PMWD framework is designed to run on a single GPU, but it is limited to grid sizes of up to 512 due to its inability to scale beyond a single device.

```python
import jax.numpy as jnp
import jaxdecomp

def pm_forces(density):
    # `density` is a 3D distributed array defined over the simulation mesh
    delta_k = jaxdecomp.fft.pfft3d(density)
    ky, kz, kx = jaxdecomp.fft.fftfreq3d(delta_k)
    kk = kx**2 + ky**2 + kz**2
    laplace_kernel = jnp.where(kk == 0, 1.0, -1.0 / kk)
    pot_k = delta_k * laplace_kernel
    forces = [-jaxdecomp.fft.ipfft3d(1j * k * pot_k) for k in [kx, ky, kz]]
    return jnp.stack(forces, axis=-1)
```

A more detailed example of an LPT simulation can be found in the [jaxdecomp_lpt example](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/blob/main/examples/lpt_nbody_demo.py).


![Visualization of an LPT density field at z=0 for a 2048³ grid generated using `jaxDecomp`.](assets/LPT_density_field_z0_2048.png){ width=65% }


# References
