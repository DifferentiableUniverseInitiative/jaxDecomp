import jax
from jax import numpy as jnp
from jax.lib import xla_client

import jaxdecomp
from jaxdecomp._src import _jaxdecomp
from jaxdecomp._src.fft_utils import FORWARD_FFTs
from jaxdecomp._src.spmd_ops import get_pdims_from_mesh, get_pencil_type

FftType = xla_client.FftType


def get_transpose_order(fft_type, mesh=None) -> tuple[int, int, int]:

  # TODO this should no longer use the global mesh
  # the parameter transpose_axis_contiguous_2 should be removed after benchmarking
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
            "only complex FFTs are currently supported through pfft.")

  pencil_type = get_pencil_type(mesh)
  match fft_type:
    case xla_client.FftType.FFT:
      # FFT is X to Y to Z so Z-Pencil is returned
      # Except if we are doing a YZ slab in which case we return a Y-Pencil
      match pencil_type:
        case _jaxdecomp.SLAB_YZ:
          if jaxdecomp.config.transpose_axis_contiguous_2:
            transpose_shape = (1, 2, 0)
          else:
            transpose_shape = (2, 0, 1)
        case _jaxdecomp.SLAB_XY | _jaxdecomp.PENCILS:
          transpose_shape = (1, 2, 0)
        case _jaxdecomp.NO_DECOMP:
          transpose_shape = (0, 1, 2)
        case _:
          raise TypeError("Unknown pencil type")
      print(f"transpose_shape is {transpose_shape}")
    case xla_client.FftType.IFFT:
      # IFFT is Z to X to Y so X-Pencil is returned
      # In YZ slab case we only need one transposition back to get the X-Pencil
      match pencil_type:
        case _jaxdecomp.SLAB_YZ:
          if jaxdecomp.config.transpose_axis_contiguous_2:
            transpose_shape = (2, 0, 1)
          else:
            transpose_shape = (1, 2, 0)
        case _jaxdecomp.SLAB_XY | _jaxdecomp.PENCILS:
          transpose_shape = (2, 0, 1)
        case _jaxdecomp.NO_DECOMP:
          transpose_shape = (0, 1, 2)
        case _:
          raise TypeError("Unknown pencil type")
    case _:
      raise TypeError("only complex FFTs are currently supported through pfft.")

  return transpose_shape


def get_lowering_args(fft_type, global_shape, mesh):
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
  global_shape = tuple([global_shape[i] for i in transpose_back_shape])

  return pdims, global_shape


def get_output_specs(fft_type, spec, mesh=None, backend='JAX'):
  pencil_type = get_pencil_type(mesh)
  # transposed_specs = None
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
    print(f"transposed_specs is {transposed_specs}")
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

      case xla_client.FftType.IFFT | xla_client.FftType.IRFFT:
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


def get_axis_names(fft_type, specs):

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


def get_axis_names_from_kind(kind: str, input_spec):
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
