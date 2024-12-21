from functools import partial
from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit, lax
from jax._src import dtypes
from jax._src.typing import Array, ArrayLike

from jaxdecomp._src.cudecomp.fft import pfft as _cudecomp_pfft
from jaxdecomp._src.fft_utils import FftType
from jaxdecomp._src.jax import fftfreq as _fftfreq
from jaxdecomp._src.jax.fft import pfft as _jax_pfft

Shape = Sequence[int]

__all__ = [
    "pfft3d",
    "pifft3d",
]


def _str_to_fft_type(s: str) -> FftType | int:
  """
    Convert a string to an FFT type enum.

    Parameters
    ----------
    s : str
        String representation of FFT type.

    Returns
    -------
    FftType
        Corresponding FFT type enum.

    Raises
    ------
    ValueError
        If the string `s` does not match known FFT types.
    """
  if s in ("fft", "FFT"):
    return FftType.FFT
  elif s in ("ifft", "IFFT"):
    return FftType.IFFT
  elif s in ("rfft", "RFFT"):
    return FftType.RFFT
  elif s in ("irfft", "IRFFT"):
    return FftType.IRFFT
  else:
    raise ValueError(f"Unknown FFT type '{s}'")


def _fft_norm(s: Array, func_name: str, norm: Optional[str]) -> Array:
  """
    Compute the normalization factor for FFT operations.

    Parameters
    ----------
    s : Array
        Shape of the input array.
    func_name : str
        Name of the FFT function ("fft" or "ifft").
    norm : str
        Type of normalization ("backward", "ortho", or "forward").

    Returns
    -------
    Array
        Normalization factor.

    Raises
    ------
    ValueError
        If an invalid norm value is provided.
    """
  if norm == "backward":
    return 1 / jnp.prod(s) if func_name.startswith("i") else jnp.array(1)
  elif norm == "ortho":
    return (1 / jnp.sqrt(jnp.prod(s)))
  elif norm == "forward":
    return jnp.array(1) if func_name.startswith("i") else 1 / jnp.prod(s)
  raise ValueError(
      f'Invalid norm value {norm}; should be "backward", "ortho" or "forward".')


@partial(jit, static_argnums=(0, 1, 3, 4))
def _do_pfft(func_name: str,
             fft_type: FftType,
             arr: Array,
             norm: Optional[str],
             backend: str = "JAX") -> Array:
  """
    Perform 3D FFT or inverse 3D FFT on the input array.

    Parameters
    ----------
    func_name : str
        Name of the FFT function ("fft" or "ifft").
    fft_type : FftType
        Type of FFT operation.
    arr : Array
        Input array to transform.
    norm : Optional[str]
        Type of normalization ("backward", "ortho", or "forward").
    backend : str, optional
        Backend to use ("JAX" or "cudecomp"), by default "JAX".

    Returns
    -------
    Array
        Transformed array after FFT or inverse FFT.
    """
  if isinstance(fft_type, str):
    typ = _str_to_fft_type(fft_type)
  elif isinstance(fft_type, FftType):  # type: ignore
    typ = fft_type
  else:
    raise TypeError(f"Unknown FFT type value '{fft_type}'")

  match typ:
    case FftType.FFT | FftType.IFFT:
      arr = lax.convert_element_type(arr,
                                     dtypes.to_complex_dtype(dtypes.dtype(arr)))
    case FftType.RFFT | FftType.IRFFT:
      raise ValueError("Not implemented wait (SOON)")

  if backend.lower() == "cudecomp":
    transformed = _cudecomp_pfft(arr, typ)
  elif backend.lower() == "jax":
    transformed = _jax_pfft(arr, typ)
  else:
    raise ValueError(f"Unknown backend value '{backend}'")

  transformed *= _fft_norm(
      jnp.array(arr.shape, dtype=transformed.dtype), func_name, norm)
  return transformed


def pfft3d(a: ArrayLike,
           norm: Optional[str] = "backward",
           backend: str = "JAX") -> Array:
  """
    Perform 3D FFT on the input array.

    Note
    ----
    The returned array is transposed compared to the input array. If the input
    is of shape (X, Y, Z), the output will be in the shape (Y, Z, X).

    Parameters
    ----------
    a : ArrayLike
        Input array to transform.
    norm : Optional[str], optional
        Type of normalization ("backward", "ortho", or "forward"), by default "backward".
    backend : str, optional
        Backend to use ("JAX" or "cudecomp"), by default "JAX".

    Returns
    -------
    Array
        Transformed array after 3D FFT.

    Example
    -------
    >>> import jax
    >>> jax.distributed.initialize()
    >>> rank = jax.process_index()
    >>> from jax.experimental import mesh_utils
    >>> from jax.sharding import Mesh, NamedSharding
    >>> from jax.sharding import PartitionSpec as P
    >>> global_shape = (16, 16, 16)
    >>> pdims = (4, 4)
    >>> local_shape = (global_shape[0] // pdims[1], global_shape[1] // pdims[0], global_shape[2])
    >>> devices = mesh_utils.create_device_mesh(pdims)
    >>> mesh = Mesh(devices.T, axis_names=('z', 'y'))
    >>> sharding = NamedSharding(mesh, P('z', 'y'))
    >>> global_array = jax.make_array_from_callback(global_shape, sharding, lambda _: jax.random.normal(jax.random.PRNGKey(rank), local_shape))
    >>> k_array = pfft3d(global_array)
    """
  return _do_pfft("fft", FftType.FFT, a, norm=norm, backend=backend)


def pifft3d(a: ArrayLike,
            norm: Optional[str] = "backward",
            backend: str = "JAX") -> Array:
  """
    Perform inverse 3D FFT on the input array.

    Note
    ----
    The returned array will have its shape restored back to (X, Y, Z) after the inverse FFT.

    Parameters
    ----------
    a : ArrayLike
        Input array to transform.
    norm : Optional[str], optional
        Type of normalization ("backward", "ortho", or "forward"), by default "backward".
    backend : str, optional
        Backend to use ("JAX" or "cudecomp"), by default "JAX".

    Returns
    -------
    Array
        Transformed array after inverse 3D FFT.

    Example
    -------
    >>> k_array = pfft3d(global_array)
    >>> original_array = pifft3d(k_array)
    """
  return _do_pfft("ifft", FftType.IFFT, a, norm=norm, backend=backend)


def fftfreq3d(array: ArrayLike, d: float = 1.0) -> Array:
  """
    Compute the 3D FFT frequency vectors.

    Note
    ----
    The input array must be in the frequency domain, meaning it must be complex.
    The order of the frequency vectors is always X, Y, Z.

    Parameters
    ----------
    array : ArrayLike
        Input array in the frequency domain.
    d : float, optional
        Sample spacing (default is 1.0).

    Returns
    -------
    Array
        3D FFT frequency vectors.

    Raises
    ------
    ValueError
        If the input array is not complex.

    Example
    -------
    >>> k_array = pfft3d(global_array)
    >>> kvec = fftfreq3d(k_array)
    """
  assert jnp.iscomplexobj(
      array), "The input array must be complex for FFT frequency computation."
  return _fftfreq.fftfreq3d(array, d=d)


def rfftfreq3d(array: ArrayLike, d: float = 1.0) -> Array:
  """
    Compute the 3D real FFT frequency vectors.

    Note
    ----
    The input array must be in the frequency domain, meaning it must be complex.
    The order of the frequency vectors is always X, Y, Z.

    Parameters
    ----------
    array : ArrayLike
        Input array in the frequency domain.
    d : float, optional
        Sample spacing (default is 1.0).

    Returns
    -------
    Array
        3D real FFT frequency vectors.

    Raises
    ------
    ValueError
        If the input array is not complex.

    Example
    -------
    >>> k_array = pfft3d(global_array)
    >>> kvec = rfftfreq3d(k_array)
    """
  assert jnp.iscomplexobj(
      array
  ), "The input array must be complex for real FFT frequency computation."
  return _fftfreq.rfftfreq3d(array, d=d)
