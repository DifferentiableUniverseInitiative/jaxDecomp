from collections.abc import Hashable
from typing import Any, Optional

import jax
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxdecomplib import _jaxdecomp

import jaxdecomp
from jaxdecomp._src.fft_utils import FftType
from jaxdecomp.typing import GdimsType, PdimsType, TransposablePdimsType

Specs = Any
AxisName = Hashable


def get_axis_size(sharding: NamedSharding, index: int) -> int:
    """Returns the size of the axis for a given sharding spec.

    Args:
        sharding: The sharding specification (PartitionSpec).
        index: The index of the axis.

    Returns:
        The size of the axis (int).
    """
    axis_name = sharding.spec[index]
    if axis_name is None:
        return 1
    else:
        return sharding.mesh.shape[sharding.spec[index]]


def get_pdims_from_sharding(sharding: NamedSharding) -> TransposablePdimsType:
    """Returns the processor dimensions from a sharding specification.

    Args:
        sharding: The sharding specification (PartitionSpec).

    Returns:
        A tuple of processor dimensions (Tuple[int, ...]).
    """
    return tuple([get_axis_size(sharding, i) for i in range(len(sharding.spec))])  # type: ignore


def get_pdims_from_mesh(mesh: Optional[Mesh]) -> PdimsType:
    """Returns the processor dimensions from the device mesh.

    Args:
        mesh: The device mesh (Mesh).

    Returns:
        A tuple of processor dimensions (Tuple[int, int]).
    """
    if mesh is None or mesh.empty:
        pdims = (1, 1)
    else:
        pdims = mesh.devices.shape[::-1]
        assert len(pdims) == 2

    return pdims


def get_pencil_type(mesh: Mesh) -> str:
    if mesh.empty:
        pdims = (1, 1)
    else:
        pdims = mesh.devices.shape
        if len(pdims) == 1:
            pdims = (1,) + pdims

        if len(pdims) != 2:
            raise ValueError('Only one or two-dimensional meshes are supported.')

    return get_pencil_type_from_pdims(pdims)


def get_pencil_type_from_pdims(pdims) -> str:
    if len(pdims) != 2:
        raise ValueError('Only one or two-dimensional meshes are supported.')

    if pdims == (1, 1) or pdims is None:
        return _jaxdecomp.NO_DECOMP
    elif pdims[0] == 1:
        return _jaxdecomp.SLAB_YZ
    elif pdims[1] == 1:
        return _jaxdecomp.SLAB_XY
    else:
        return _jaxdecomp.PENCILS


def get_axis_names_from_mesh(mesh: Mesh) -> tuple[str, str]:
    return mesh.axis_names + (None,) * (2 - len(mesh.axis_names))


def get_pdims_from_axis_names(x_axis_name: AxisName, y_axis_name: AxisName) -> PdimsType:
    x_size = 1 if x_axis_name is None else lax.psum(1, x_axis_name)
    y_size = 1 if y_axis_name is None else lax.psum(1, y_axis_name)

    return (x_size, y_size)


def get_pencil_type_from_axis_names(x_axis_name: AxisName, y_axis_name: AxisName) -> str:
    pdims = get_pdims_from_axis_names(x_axis_name, y_axis_name)
    return get_pencil_type_from_pdims(pdims)


def validate_spec_matches_mesh(spec, mesh: Mesh):
    """Validate that the sharding spec is equivalent to the mesh's natural partitioning.

    Equivalence rules (checked position by position):
    - spec[i] == mesh.axis_names[i]  ->  direct match
    - spec[i] is None and mesh.shape[mesh.axis_names[i]] == 1  ->  OK (size-1 axis = unsharded)
    - spec[i] is None and i >= len(mesh.axis_names)  ->  OK (trailing unsharded dim)
    - Otherwise  ->  ERROR (wrong order or wrong axis name)
    """
    spec = spec + (None,) * (3 - len(spec))
    mesh_names = mesh.axis_names

    for i in range(min(len(spec), 3)):
        if spec[i] is not None:
            if i >= len(mesh_names) or spec[i] != mesh_names[i]:
                raise ValueError(
                    f"Spec {spec} does not match mesh axis names {mesh_names}. "
                    f"At position {i}: spec has '{spec[i]}' but mesh expects "
                    f"'{mesh_names[i] if i < len(mesh_names) else 'nothing (trailing)'}'. "
                    "Ensure the array is sharded with a spec matching the mesh axis order."
                )
        else:
            if i < len(mesh_names) and mesh.shape[mesh_names[i]] > 1:
                raise ValueError(
                    f'Spec {spec} does not match mesh axis names {mesh_names}. '
                    f"At position {i}: spec has None but mesh axis '{mesh_names[i]}' "
                    f'has size {mesh.shape[mesh_names[i]]} (> 1). '
                    'An unsharded dimension cannot correspond to a distributed mesh axis.'
                )


def get_transpose_order(fft_type: FftType) -> tuple[int, int, int]:
    """Returns the transpose order based on the FFT type and mesh configuration.

    Args:
        fft_type: The type of FFT (FftType).
        mesh: The device mesh (Optional[Mesh]).
        spec: The PartitionSpec (Optional). If provided with mesh, uses spec-based pencil detection.

    Returns:
        A tuple representing the transpose order (Tuple[int, int, int]).

    Raises:
        TypeError: If an unknown FFT type or pencil type is encountered.
    """
    if not jaxdecomp.config.transpose_axis_contiguous:
        return (0, 1, 2)

    match fft_type:
        case FftType.FFT | FftType.RFFT:
            return (1, 2, 0)
        case FftType.IFFT | FftType.IRFFT:
            return (2, 0, 1)
        case _:
            raise TypeError('Only complex FFTs are currently supported through pfft.')


def get_lowering_args(fft_type: FftType, global_shape: GdimsType, mesh: Mesh) -> tuple[PdimsType, GdimsType]:
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
            case lax.FftType.FFT:
                if pencil_type == _jaxdecomp.SLAB_XY:
                    transpose_back_shape = (1, 2, 0)
                    pdims = pdims[::-1]
                else:
                    transpose_back_shape = (0, 1, 2)
            case lax.FftType.IFFT:
                if pencil_type == _jaxdecomp.SLAB_XY:
                    transpose_back_shape = (0, 1, 2)
                    pdims = pdims[::-1]
                elif pencil_type == _jaxdecomp.SLAB_YZ:
                    transpose_back_shape = (2, 0, 1)
                else:
                    transpose_back_shape = (2, 0, 1)
            case _:
                raise TypeError('only complex FFTs are currently supported through pfft.')
    else:
        transpose_back_shape = (0, 1, 2)
    # Make sure to get back the original shape of the X-Pencil
    global_shape = (
        global_shape[transpose_back_shape[0]],
        global_shape[transpose_back_shape[1]],
        global_shape[transpose_back_shape[2]],
    )
    assert len(pdims) == 2

    return pdims, global_shape


def get_fft_output_sharding(fft_sharding):
    spec = fft_sharding.spec
    mesh = fft_sharding.mesh
    pencil_type = get_pencil_type(mesh)
    out_specs = get_output_specs(FftType.FFT, pencil_type, spec)

    return NamedSharding(mesh, P(*out_specs))


def get_output_specs(fft_type: FftType, pencil_type: str, spec: P, backend: str = 'JAX') -> tuple[Optional[int], ...]:
    """Returns the output specs based on FFT type, pencil type, and spec.

    Args:
        fft_type: The type of FFT (FftType).
        pencil_type: The pencil decomposition type (str).
        spec: The input specs (PartitionSpec).
        backend: The backend to use (str).

    Returns:
        The transposed output specs (TransposablePdimsType).

    Raises:
        TypeError: If an unknown FFT or pencil type is encountered.
        ValueError: If invalid sharding or distributed specs are provided.
    """
    spec = spec + (None,) * (3 - len(spec))
    if jaxdecomp.config.transpose_axis_contiguous:
        match pencil_type:
            case _jaxdecomp.SLAB_XY:
                transposed_specs = (spec[1], spec[0], spec[2])
            case _jaxdecomp.SLAB_YZ:
                transposed_specs = (spec[2], spec[1], spec[0])
            case _jaxdecomp.PENCILS:
                transposed_specs = spec
            case _:
                raise TypeError('Unknown pencil type')
    else:

        def is_distributed(x):
            if type(x) is str:
                # Cannot check if the axis is distributed at compile time if using shardy
                return False
            return x is not None and x != 1

        match fft_type:
            case FftType.FFT | FftType.RFFT:
                if is_distributed(spec[2]):
                    raise ValueError(
                        'Distributed FFTs with non-contiguous axes does not support a third distributed axis'
                        f'Make sure that the device mesh is created with a 1D or 2D mesh, got spec {spec}'
                    )
                match pencil_type:
                    case _jaxdecomp.SLAB_XY:
                        if backend == 'cudecomp':
                            transposed_specs = (spec[1], spec[0], spec[2])
                        else:
                            transposed_specs = (spec[2], spec[1], spec[0])
                    case _jaxdecomp.SLAB_YZ:
                        transposed_specs = (spec[0], spec[2], spec[1])
                    case _jaxdecomp.PENCILS:
                        transposed_specs = (spec[2], spec[0], spec[1])
                    case _:
                        raise TypeError('Unknown pencil type')

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
                        if backend == 'cudecomp':
                            transposed_specs = (spec[1], spec[0], spec[2])
                        else:
                            transposed_specs = (spec[2], spec[1], spec[0])
                    case _jaxdecomp.SLAB_YZ:
                        if is_distributed(spec[0]) or is_distributed(spec[1]):
                            raise ValueError(
                                'Distributed IFFT with a YZ slab (only X axis distributed) is not compatible with the current sharding'
                                f'got {spec} expected {(None, None, jax.device_count())}'
                                'Make sure that you use IFFT on the output of a distributed FFT'
                                'or create it with a NamedSharding with a PartitionSpec of (None, None, jax.device_count())'
                            )
                        transposed_specs = (spec[0], spec[2], spec[1])
                    case _jaxdecomp.PENCILS:
                        if is_distributed(spec[0]):
                            raise ValueError(
                                'Distributed IFFT with a PENCILS decomposition (Both Y and Z distributed)'
                                ' is not compatible with the current sharding'
                                f'got {spec} expected {(None, jax.device_count() // 2, jax.device_count() // (jax.device_count() // 2))} '
                                'or any other 2D mesh'
                                'Make sure that you use IFFT on the output of a distributed FFT'
                                'or create it with a NamedSharding with a PartitionSpec of (None, 2 , 2) or any other 2D mesh'
                            )
                        transposed_specs = (spec[1], spec[2], spec[0])
                    case _:
                        raise TypeError('Unknown pencil type')

            case _:
                raise TypeError('only complex FFTs are currently supported through pfft.')

    return transposed_specs
