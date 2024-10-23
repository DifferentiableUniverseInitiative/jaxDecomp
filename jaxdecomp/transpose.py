from jaxtyping import Array

from jaxdecomp._src.cudecomp.transpose import transpose as _cudecomp_transpose
from jaxdecomp._src.jax.transpose import transpose as _jax_transpose


def transposeXtoY(x: Array, backend: str = 'JAX') -> Array:
  if backend.lower() == 'jax':
    return _jax_transpose(x, kind='x_y')
  elif backend.lower() == 'cudecomp':
    return _cudecomp_transpose(x, kind='x_y')
  else:
    raise ValueError(f"Invalid backend: {backend}")


def transposeYtoX(x: Array, backend: str = 'JAX') -> Array:
  if backend.lower() == 'jax':
    return _jax_transpose(x, kind='y_x')
  elif backend.lower() == 'cudecomp':
    return _cudecomp_transpose(x, kind='y_x')
  else:
    raise ValueError(f"Invalid backend: {backend}")


def transposeYtoZ(x: Array, backend: str = 'JAX') -> Array:
  if backend.lower() == 'jax':
    return _jax_transpose(x, kind='y_z')
  elif backend.lower() == 'cudecomp':
    return _cudecomp_transpose(x, kind='y_z')
  else:
    raise ValueError(f"Invalid backend: {backend}")


def transposeZtoY(x: Array, backend: str = 'JAX') -> Array:
  if backend.lower() == 'jax':
    return _jax_transpose(x, kind='z_y')
  elif backend.lower() == 'cudecomp':
    return _cudecomp_transpose(x, kind='z_y')
  else:
    raise ValueError(f"Invalid backend: {backend}")


def transposeXtoZ(x: Array, backend: str = 'JAX') -> Array:
  if backend.lower() == 'jax':
    return _jax_transpose(x, kind='x_z')
  elif backend.lower() == 'cudecomp':
    raise NotImplementedError("Cudecomp does not support x_z transpose")
  else:
    raise ValueError(f"Invalid backend: {backend}")


def transposeZtoX(x: Array, backend: str = 'JAX') -> Array:
  if backend.lower() == 'jax':
    return _jax_transpose(x, kind='z_x')
  elif backend.lower() == 'cudecomp':
    raise NotImplementedError("Cudecomp does not support z_x transpose")
  else:
    raise ValueError(f"Invalid backend: {backend}")
