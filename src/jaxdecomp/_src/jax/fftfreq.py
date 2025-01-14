from functools import partial
from typing import Tuple

import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental.shard_alike import shard_alike
from jaxtyping import Array

FftType = lax.FftType


@partial(jax.jit, static_argnums=(1,))
def fftfreq3d(k_array: Array, d: float = 1.0) -> Tuple[Array, Array, Array]:
    """
    Computes the 3D FFT frequencies for a given array, assuming a Z-pencil configuration.

    Parameters
    ----------
    k_array : Array
        Input array for which the FFT frequencies are to be computed.
    d : float, optional
        Sampling interval in any direction. Defaults to 1.0.

    Returns
    -------
    Tuple[Array, Array, Array]
        The frequencies corresponding to the Z, Y, and X axes, respectively.
    """
    if jnp.iscomplexobj(k_array):
        dtype = jnp.float32 if k_array.dtype == jnp.complex64 else jnp.float64
    else:
        dtype = k_array.dtype

    # Compute the FFT frequencies for each axis
    ky = jnp.fft.fftfreq(k_array.shape[0], d=d, dtype=dtype) * 2 * jnp.pi
    kx = jnp.fft.fftfreq(k_array.shape[1], d=d, dtype=dtype) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(k_array.shape[2], d=d, dtype=dtype) * 2 * jnp.pi

    k_array_structure = jax.tree.structure(k_array)
    kx = jax.tree.unflatten(k_array_structure, (kx,))
    ky = jax.tree.unflatten(k_array_structure, (ky,))
    kz = jax.tree.unflatten(k_array_structure, (kz,))

    # Ensure frequencies are sharded similarly to the input array
    ky, _ = shard_alike(ky, k_array[:, 0, 0])
    kx, _ = shard_alike(kx, k_array[0, :, 0])
    kz, _ = shard_alike(kz, k_array[0, 0, :])

    # Reshape the frequencies to match the input array's dimensionality
    ky = ky.reshape([-1, 1, 1])
    kx = kx.reshape([1, -1, 1])
    kz = kz.reshape([1, 1, -1])

    return kz, ky, kx


@partial(jax.jit, static_argnums=(1,))
def rfftfreq3d(k_array: Array, d: float = 1.0) -> Tuple[Array, Array, Array]:
    """
    Computes the 3D FFT frequencies for a real input array, assuming a Z-pencil configuration.
    The FFT is computed for the real input on the X axis using rfft.

    Parameters
    ----------
    k_array : Array
        Input array for which the rFFT frequencies are to be computed.
    d : float, optional
        Sampling interval in any direction. Defaults to 1.0.

    Returns
    -------
    Tuple[Array, Array, Array]
        The frequencies corresponding to the Z, Y, and X axes, respectively.
    """
    if jnp.iscomplexobj(k_array):
        dtype = jnp.float32 if k_array.dtype == jnp.complex64 else jnp.float64
    else:
        dtype = k_array.dtype

    # Compute the FFT frequencies for each axis
    ky = jnp.fft.fftfreq(k_array.shape[0], d=d, dtype=dtype) * 2 * jnp.pi
    kx = jnp.fft.rfftfreq(k_array.shape[1], d=d, dtype=dtype) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(k_array.shape[2], d=d, dtype=dtype) * 2 * jnp.pi


    k_array_structure = jax.tree.structure(k_array)
    kx = jax.tree.unflatten(k_array_structure, (kx,))
    ky = jax.tree.unflatten(k_array_structure, (ky,))
    kz = jax.tree.unflatten(k_array_structure, (kz,))

    # Ensure frequencies are sharded similarly to the input array
    ky, _ = shard_alike(ky, k_array[:, 0, 0])
    kx, _ = shard_alike(kx, k_array[0, :, 0])
    kz, _ = shard_alike(kz, k_array[0, 0, :])

    # Reshape the frequencies to match the input array's dimensionality
    ky = ky.reshape([-1, 1, 1])
    kx = kx.reshape([1, -1, 1])
    kz = kz.reshape([1, 1, -1])

    return kz, ky, kx
