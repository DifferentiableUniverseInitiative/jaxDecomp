from math import prod
from typing import TypeAlias

from jax import lax
from jax import numpy as jnp
from jaxtyping import Array

FftType: TypeAlias = lax.FftType

FORWARD_FFTs = {FftType.FFT, FftType.RFFT}
INVERSE_FFTs = {FftType.IFFT, FftType.IRFFT}


def ADJOINT(fft_type: FftType) -> FftType:
    """Returns the adjoint (inverse) of the given FFT type.

    Args:
        fft_type: The type of FFT (FftType).

    Returns:
        The adjoint (inverse) FFT type (FftType).

    Raises:
        ValueError: If an unknown FFT type is provided.
    """
    match fft_type:
        case FftType.FFT:
            return FftType.IFFT
        case FftType.IFFT:
            return FftType.FFT
        case FftType.RFFT:
            return FftType.IRFFT
        case FftType.IRFFT:
            return FftType.RFFT
        case _:
            raise ValueError(f"Unknown FFT type '{fft_type}'")


def COMPLEX(fft_type: FftType) -> FftType:
    """Returns the complex equivalent of the given FFT type.

    Args:
        fft_type: The type of FFT (FftType).

    Returns:
        The complex FFT type (FftType).

    Raises:
        ValueError: If an unknown FFT type is provided.
    """
    match fft_type:
        case FftType.RFFT | FftType.FFT:
            return FftType.FFT
        case FftType.IRFFT | FftType.IFFT:
            return FftType.IFFT
        case _:
            raise ValueError(f"Unknown FFT type '{fft_type}'")


def _un_normalize_fft(s: tuple[int, ...], fft_type: FftType) -> Array:
    """Computes the un-normalization factor for the FFT.

    Args:
        s: Shape of the array (Tuple[int, ...]).
        fft_type: The type of FFT (FftType).

    Returns:
        The un-normalization factor (Array).
    """
    if fft_type in FORWARD_FFTs:
        return jnp.array(1)
    else:
        return jnp.array(prod(s))


def fftn(a: Array, fft_type: FftType, adjoint: bool) -> Array:
    """Performs an n-dimensional FFT on the input array.

    Args:
        a: Input array (Array).
        fft_type: The type of FFT (FftType).
        adjoint: Whether to apply the adjoint (inverse) FFT (bool).

    Returns:
        The transformed array (Array).

    Raises:
        ValueError: If an unknown FFT type is provided.
    """
    if fft_type in FORWARD_FFTs:
        axes = tuple(range(0, 3))
    else:
        axes = tuple(range(2, -1, -1))

    if adjoint:
        fft_type = ADJOINT(fft_type)

    if fft_type == FftType.FFT:
        a = jnp.fft.fftn(a, axes=axes)
    elif fft_type == FftType.IFFT:
        a = jnp.fft.ifftn(a, axes=axes)
    elif fft_type == FftType.RFFT:
        a = jnp.fft.rfftn(a, axes=axes)
    elif fft_type == FftType.IRFFT:
        a = jnp.fft.irfftn(a, axes=axes)
    else:
        raise ValueError(f"Unknown FFT type '{fft_type}'")

    s = a.shape
    a *= _un_normalize_fft(s, fft_type)

    return a


def fft(a: Array, fft_type: FftType, axis: int, adjoint: bool) -> Array:
    """Performs a 1-dimensional FFT along the specified axis of the input array.

    Args:
        a: Input array (Array).
        fft_type: The type of FFT (FftType).
        axis: The axis along which to compute the FFT (int).
        adjoint: Whether to apply the adjoint (inverse) FFT (bool).

    Returns:
        The transformed array (Array).

    Raises:
        ValueError: If an unknown FFT type is provided.
    """
    if adjoint:
        fft_type = ADJOINT(fft_type)

    if fft_type == FftType.FFT:
        a = jnp.fft.fft(a, axis=axis)
    elif fft_type == FftType.IFFT:
        a = jnp.fft.ifft(a, axis=axis)
    elif fft_type == FftType.RFFT:
        a = jnp.fft.rfft(a, axis=axis)
    elif fft_type == FftType.IRFFT:
        a = jnp.fft.irfft(a, axis=axis)
    else:
        raise ValueError(f"Unknown FFT type '{fft_type}'")

    s = (a.shape[axis],)
    a *= _un_normalize_fft(s, fft_type)

    return a


def fft2(a: Array, fft_type: FftType, axes: tuple[int, int], adjoint: bool) -> Array:
    """Performs a 2-dimensional FFT along the specified axes of the input array.

    Args:
        a: Input array (Array).
        fft_type: The type of FFT (FftType).
        axes: The axes along which to compute the FFT (Tuple[int, int]).
        adjoint: Whether to apply the adjoint (inverse) FFT (bool).

    Returns:
        The transformed array (Array).

    Raises:
        ValueError: If an unknown FFT type is provided.
    """
    if adjoint:
        fft_type = ADJOINT(fft_type)

    if fft_type == FftType.FFT:
        a = jnp.fft.fft2(a, axes=axes)
    elif fft_type == FftType.IFFT:
        a = jnp.fft.ifft2(a, axes=axes)
    elif fft_type == FftType.RFFT:
        a = jnp.fft.rfft2(a, axes=axes)
    elif fft_type == FftType.IRFFT:
        a = jnp.fft.irfft2(a, axes=axes)
    else:
        raise ValueError(f"Unknown FFT type '{fft_type}'")

    s = tuple(a.shape[i] for i in axes)
    a *= _un_normalize_fft(s, fft_type)

    return a
