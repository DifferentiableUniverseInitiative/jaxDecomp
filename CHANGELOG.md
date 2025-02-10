# Change log

## jaxdecomp 0.2.0

* Changes
  * jaxDecomp works without MPI and using only JAX as backend
  * with mesh is no longer required (will be deprecated by JAX)
  * Added support for fftfreq
  * Added testing for all functions
  * Added static typing and checked with mypy

## jaxdecomp 0.1.0

* Changes
  * Fixed bug with Halo
  * Added joss paper


## jaxdecomp 0.0.1

* Changes
  * New version compatible with JAX 0.4.30
  * jaxDecomp now works in a multi-host environment
  * Added custom partitioning for FFTs
  * Added custom partitioning for halo exchange
  * Added custom partitioning for slice_pad and slice_unpad
  * Add example for multi-host FFTs in `examples/jaxdecomp_lpt.py`


## jaxdecomp 0.0.1rc2
* Changes
  * Added utility to run autotuning


## jaxdecomp 0.0.1rc1 (Nov. 25th 2022)

Initial pre-release, include support for parallel ffts and halo exchange
