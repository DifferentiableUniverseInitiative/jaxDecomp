---
title: 'jaxDecomp : JAX Library for 3D Domain Decomposition and Parallel FFTs'
tags:
  - Python
  - Hpc
  - Cuda
  - Jax
  - FFT
  - Simulations
authors:
  - name: Wassim KABALAN
    orcid: 0009-0001-6501-4564
    affiliation: 1
  - name: François Lanusse
    affiliation: 2
  - name: Alexandre Boucaud
    affiliation: 1
  - name: Eric Aubourg
    affiliation: 3
affiliations:
 - name: Université Paris Cité, CNRS, Astroparticule et Cosmologie, F-75013 Paris, France
   index: 1
 - name: Université Paris-Saclay, Université Paris Cité, CEA, CNRS, AIM, 91191, Gif-sur-Yvette, France
   index: 2
date: 26 June 2024
bibliography: paper.bib

---


# Summary

Cosmological simulations are a key tool to help us understand the distribution of galaxies and dark matter in the universe. Differentiable simulations give access to the gradients which significantly accelerate the inference process. Fast Particle Mesh (PM) simulations are a very good candidate due to their speed and simplicity and thus differentiability. However, entering the exascale era, simulation sizes are surpassing the maximum available memory even for the high-end HPC GPUs. For that, a multi-node distributed Particle Mesh simulation is needed to be truly able to simulate the large cosmological volumes. The only step that requires communications in the fast PM simulations is the fast Fourier transform (FFT). There are a few implementations of distributed FFTs coming from the computer science community, like [@2DECOMP&FFT] that allows distributed FFTs on CPUs and the GPU version [@cuDecomp] that uses NVIDIA Collective Communication Library (NCCL) for the communication. However, these libraries are not integrated with the differentiable simulation libraries like JAX. To address this, we introduce `jaxDecomp`, a `JAX` library based on `cuDecomp` that efficiently decomposes the simulation data into 2D slices (pencils) to facilitate multi-node parallel Fast Fourier Transforms (FFTs) and halo exchanges, leveraging the power of compiled code directly within `JAX` code. This library will enable the large-scale distribution of simulations on High Performance Computing (HPC) clusters and will seamlessly integrate with existing open-source simulation codes like `JaxPM` or [@pmwd].

# Statement of Need

Particle mesh simulations are essential for cosmological data analysis, particularly in full field inference. They simulate the large-scale structure of the universe and generate the likelihood of the data given the cosmological parameters. Differentiable simulations unlock advanced sampling techniques like Hamiltonian Monte Carlo (HMC) and variational inference. To maximize the potential of particle mesh simulations, it is crucial to use a very fine grid, achieving a high resolution of small-scale structures and a power spectrum close to that of hydrodynamical simulations.

Full field inference, based on Bayesian hierarchical models, allows for the inference of cosmological parameters from galaxy survey data. This method utilizes all available data, rather than just estimated two-point correlation functions, this requires fast simulations of sizable fractions of the universe. Particle mesh simulations play a pivotal role in this process by generating the likelihood of the data given the cosmological parameters.

To perform these simulations efficiently on modern HPC clusters, distributed FFTs are required. `jaxDecomp` addresses this need by distributing FFTs across multiple GPUs and nodes, fully compatible with JAX. This allows for simple Python API usage while benefiting from the performance of compiled code on single or multiple GPUs.

# Implementation

## Distributed FFTs

The implementation of `jaxDecomp` focuses on 2D decomposition for parallel FFTs and efficient halo exchange. The process begins by creating a 2D grid of GPUs using the JAX API, followed by the creation of particles on this grid. The steps for performing FFTs and their inverse are as follows:

```python
FFT1D_X(particles)
y_pencil = TransposeXtoY(particles)
FFT1D_Y(y_pencil)
z_pencil = TransposeYtoZ(y_pencil)
FFT1D_Z(z_pencil)
```

And inverse FFTs goes the other way around

```python
FFT1D_Z_inv(z_pencil)
y_pencil = TransposeZtoY(z_pencil)
FFT1D_Y_inv(y_pencil)
x_pencil = TransposeYtoX(y_pencil)
FFT1D_X_inv(x_pencil)
```
<p align="center">
  <img src="assets/fft.svg" alt="Distributed FFTs using jaxDecomp" style="width: 40%">
</div>



## Distributed Halo Exchange

During the update step, particles cannot leave the local slice (GPU). jaxDecomp facilitates a halo exchange to handle particles on the edge of slices efficiently using the NCCL library.

<p align="center">
  <img src="assets/halo-exchange.svg"  alt="Distributed Halo Exchange using jaxDecomp" style="width: 40%"/>
</p>


Key FeaturesDifferentiable: Seamlessly integrates with JAX for differentiable simulations.Scalable: Efficiently distributes FFTs across multiple GPUs and nodes.User-Friendly: Provides a simple Python API, leveraging the power of compiled code.Halo Exchange: Performs efficient halo exchange using NCCL for particle updates.AcknowledgementsWe acknowledge contributions from François Lanusse and support from the Université Paris Cité and Université Paris-Saclay. Special thanks to the developers of cuDecomp and 2DECOMP&FFT for their foundational work in distributed FFTs.References<!-- Add your references here -->yamlCopy code
---

This draft incorporates your points and fills out the required sections. You can add specific figures, benchmarks, and additional details to further complete your paper. If you need further assistance with any particular section or additional information, feel free to ask!
