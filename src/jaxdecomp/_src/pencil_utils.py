from typing import Optional, Tuple

import jax
from jax.lib import xla_client
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import jaxdecomp
from jaxdecomp._src import _jaxdecomp
from jaxdecomp._src.fft_utils import FftType, FORWARD_FFTs
from jaxdecomp._src.spmd_ops import get_pdims_from_mesh, get_pencil_type
from jaxdecomp.typing import GdimsType, PdimsType, TransposablePdimsType


def get_transpose_order(fft_type: FftType,
                        mesh: Optional[Mesh] = None) -> Tuple[int, int, int]:
  """Returns the transpose order based on the FFT type and mesh configuration.

    Args:
        fft_type: The type of FFT (FftType).
        mesh: The device mesh (Optional[Mesh]).

    Returns:
        A tuple representing the transpose order (Tuple[int, int, int]).

    Raises:
        TypeError: If an unknown FFT type or pencil type is encountered.
    """
  if not jaxdecomp.config.transpose_axis_contiguous:
    return (0, 1, 2)

  if mesh is None:
    match fft_type:
      case FftType.FFT | FftType.RFFT:
        return (1, 2, 0)
      case FftType.IFFT | FftType.IRFFT:
        return (2, 0, 1)
      case _:
        raise TypeError(
            "Only complex FFTs are currently supported through pfft.")

  pencil_type = get_pencil_type(mesh)
  match fft_type:
    case FftType.FFT:
      match pencil_type:
        case _jaxdecomp.SLAB_YZ:
          if jaxdecomp.config.transpose_axis_contiguous_2:
            return (1, 2, 0)
          else:
            return (2, 0, 1)
        case _jaxdecomp.SLAB_XY | _jaxdecomp.PENCILS:
          return (1, 2, 0)
        case _jaxdecomp.NO_DECOMP:
          return (0, 1, 2)
        case _:
          raise TypeError("Unknown pencil type")
    case FftType.IFFT:
      match pencil_type:
        case _jaxdecomp.SLAB_YZ:
          if jaxdecomp.config.transpose_axis_contiguous_2:
            return (2, 0, 1)
          else:
            return (1, 2, 0)
        case _jaxdecomp.SLAB_XY | _jaxdecomp.PENCILS:
          return (2, 0, 1)
        case _jaxdecomp.NO_DECOMP:
          return (0, 1, 2)
        case _:
          raise TypeError("Unknown pencil type")
    case _:
      raise TypeError("Only complex FFTs are currently supported through pfft.")


def get_lowering_args(fft_type: FftType, global_shape: GdimsType,
                      mesh: Mesh) -> Tuple[PdimsType, GdimsType]:
  """Returns the lowering arguments based on FFT type, global shape, and mesh.

    Args:
        fft_type: The type of FFT (FftType).
        global_shape: The global shape of the array (GdimsType).
        mesh: The device mesh (Mesh).

    Returns:
        A tuple containing pdims (processor dimensions) and the new global shape (GdimsType).

    Raises:
        TypeError: If an unknown FFT type is encountered.
    """
  pencil_type = get_pencil_type(mesh)
  pdims = get_pdims_from_mesh(mesh)

  if jaxdecomp.config.transpose_axis_contiguous:
    match fft_type:
      case xla_client.FftType.FFT:
        if pencil_type == _jaxdecomp.SLAB_XY:
          transpose_back_shape = (1, 2, 0)
          pdims = pdims[::-1]
        else:
          transpose_back_shape = (0, 1, 2)
      case xla_client.FftType.IFFT:
        if pencil_type == _jaxdecomp.SLAB_XY:
          transpose_back_shape = (0, 1, 2)
          pdims = pdims[::-1]
        elif pencil_type == _jaxdecomp.SLAB_YZ:
          if jaxdecomp.config.transpose_axis_contiguous_2:
            transpose_back_shape = (2, 0, 1)
          else:
            transpose_back_shape = (1, 2, 0)
        else:
          transpose_back_shape = (2, 0, 1)
      case _:
        raise TypeError(
            "only complex FFTs are currently supported through pfft.")
  else:
    transpose_back_shape = (0, 1, 2)
  # Make sure to get back the original shape of the X-Pencil
  global_shape = (global_shape[transpose_back_shape[0]],
                  global_shape[transpose_back_shape[1]],
                  global_shape[transpose_back_shape[2]])
  assert len(pdims) == 2

  return pdims, global_shape


def get_output_specs(
    fft_type: FftType,
    spec: P,
    mesh: Optional[Mesh] = None,
    backend: str = 'JAX') -> Tuple[Optional[int], Optional[int], Optional[int]]:
  """Returns the output specs based on FFT type, spec, and mesh.

    Args:
        fft_type: The type of FFT (FftType).
        spec: The input specs (PartitionSpec).
        mesh: The device mesh (Optional[Mesh]).
        backend: The backend to use (str).

    Returns:
        The transposed output specs (TransposablePdimsType).

    Raises:
        TypeError: If an unknown FFT or pencil type is encountered.
        ValueError: If invalid sharding or distributed specs are provided.
    """
  pencil_type = get_pencil_type(mesh)
  if jaxdecomp.config.transpose_axis_contiguous:
    match pencil_type:
      case _jaxdecomp.SLAB_XY:
        transposed_specs = (spec[1], spec[0], None)
      case _jaxdecomp.SLAB_YZ:
        if jaxdecomp.config.transpose_axis_contiguous_2:
          transposed_specs = (None, spec[1], spec[0])
        else:
          transposed_specs = (spec[1], spec[0], None)
      case _jaxdecomp.PENCILS:
        transposed_specs = spec
      case _:
        raise TypeError("Unknown pencil type")
  else:
    is_distributed = lambda x: x is not None and x != 1
    match fft_type:
      case FftType.FFT | FftType.RFFT:
        if (is_distributed(spec[2])):
          raise ValueError(
              "Distributed FFTs with non-contiguous axes does not support a third distributed axis"
              f"Make sure that the device mesh is created with a 1D or 2D mesh, got spec {spec}"
          )
        match pencil_type:
          case _jaxdecomp.SLAB_XY:
            if backend == "cudecomp":
              transposed_specs = (None, spec[0], None)
            else:
              transposed_specs = (None, None, spec[0])
          case _jaxdecomp.SLAB_YZ:
            transposed_specs = (None, None, spec[1])
          case _jaxdecomp.PENCILS:
            transposed_specs = (None, spec[0], spec[1])
          case _:
            raise TypeError("Unknown pencil type")

      case FftType.IFFT | FftType.IRFFT:
        match pencil_type:
          case _jaxdecomp.SLAB_XY:
            # if (is_distributed(spec[0]) or is_distributed(spec[2])):
            #   raise ValueError(
            #       f"Distributed IFFT with a XY slab (only Z axis distributed) is not compatible with the current sharding"
            #       f"got {spec} expected {(None , jax.device_count(), None)}"
            #       f"Make sure that you use IFFT on the output of a distributed FFT"
            #       "or create it with a NamedSharding with a PartitionSpec of (None, jax.device_count(), None)"
            #   )
            if backend == "cudecomp":
              transposed_specs = (spec[1], None, None)
            else:
              transposed_specs = (spec[2], None, None)
          case _jaxdecomp.SLAB_YZ:
            if (is_distributed(spec[0]) or is_distributed(spec[1])):
              raise ValueError(
                  f"Distributed IFFT with a YZ slab (only X axis distributed) is not compatible with the current sharding"
                  f"got {spec} expected {(None , None, jax.device_count())}"
                  f"Make sure that you use IFFT on the output of a distributed FFT"
                  "or create it with a NamedSharding with a PartitionSpec of (None, None, jax.device_count())"
              )
            transposed_specs = (None, spec[2], None)
          case _jaxdecomp.PENCILS:
            if (is_distributed(spec[0])):
              raise ValueError(
                  f"Distributed IFFT with a PENCILS decomposition (Both Y and Z distributed) is not compatible with the current sharding"
                  f"got {spec} expected {(None , jax.device_count() //2, jax.device_count() // (jax.device_count() //2) )} or any other 2D mesh"
                  f"Make sure that you use IFFT on the output of a distributed FFT"
                  "or create it with a NamedSharding with a PartitionSpec of (None, 2 , 2) or any other 2D mesh"
              )
            transposed_specs = (spec[1], spec[2], None)
          case _:
            raise TypeError("Unknown pencil type")

      case _:
        raise TypeError(
            "only complex FFTs are currently supported through pfft.")

  return transposed_specs


def get_axis_names(fft_type: FftType,
                   specs: TransposablePdimsType) -> Tuple[int, int]:
  """Returns the axis names based on the FFT type and specs.

    Args:
        fft_type: The type of FFT (FftType).
        specs: The specs for each axis (TransposablePdimsType).

    Returns:
        A tuple of axis names (Tuple[int, int]).

    Raises:
        TypeError: If an unknown pencil type is encountered.
    """
  pencil_type = get_pencil_type()

  if fft_type in FORWARD_FFTs:
    return specs[0], specs[1]

  if jaxdecomp.config.transpose_axis_contiguous:
    match pencil_type:
      case _jaxdecomp.SLAB_XY | _jaxdecomp.SLAB_YZ:
        return specs[1], specs[0]
      case _jaxdecomp.PENCILS:
        return specs[0], specs[1]
      case _:
        raise TypeError("Unknown pencil type")
  else:
    match pencil_type:
      case _jaxdecomp.SLAB_XY:
        return specs[2], specs[1]
      case _jaxdecomp.SLAB_YZ:
        return specs[0], specs[2]
      case _jaxdecomp.PENCILS:
        return specs[1], specs[2]
      case _:
        raise TypeError("Unknown pencil type")


def get_axis_names_from_kind(
    kind: str, input_spec: TransposablePdimsType) -> Tuple[int, int]:
  """Returns the axis names based on the kind and input specs.

    Args:
        kind: The kind of axis decomposition (str).
        input_spec: The input specs (TransposablePdimsType).

    Returns:
        A tuple of axis names (Tuple[int, int]).

    Raises:
        ValueError: If an invalid kind is provided.
    """
  if jaxdecomp.config.transpose_axis_contiguous:
    match kind:
      case 'x_y' | 'z_y' | 'x_z':
        return input_spec[0], input_spec[1]
      case 'y_z' | 'y_x' | 'z_x':
        return input_spec[1], input_spec[0]
      case _:
        raise ValueError("Invalid kind")
  else:
    match kind:
      case 'x_y':
        return input_spec[0], input_spec[1]
      case 'y_z':
        return input_spec[0], input_spec[2]
      case 'z_y':
        return input_spec[1], input_spec[2]
      case 'y_x':
        return input_spec[0], input_spec[2]
      case 'x_z':
        return input_spec[0], input_spec[1]
      case 'z_x':
        return input_spec[2], input_spec[1]
      case _:
        raise ValueError("Invalid kind")
