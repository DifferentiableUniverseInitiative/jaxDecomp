from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

from jaxdecomp._src.pencil_utils import get_fft_output_sharding, get_output_specs, validate_spec_matches_mesh
from jaxdecomp.fft import fftfreq3d, pfft3d, pifft3d, rfftfreq3d
from jaxdecomp.halo import halo_exchange
from jaxdecomp.transpose import (
    transposeXtoY,
    transposeXtoZ,
    transposeYtoX,
    transposeYtoZ,
    transposeZtoX,
    transposeZtoY,
)

from ._src import (
    HALO_COMM_MPI,
    HALO_COMM_MPI_BLOCKING,
    HALO_COMM_NCCL,
    HALO_COMM_NVSHMEM,
    HALO_COMM_NVSHMEM_BLOCKING,
    NO_DECOMP,
    PENCILS,
    SLAB_XY,
    SLAB_YZ,
    TRANSPOSE_COMM_MPI_A2A,
    TRANSPOSE_COMM_MPI_P2P,
    TRANSPOSE_COMM_MPI_P2P_PL,
    TRANSPOSE_COMM_NCCL,
    TRANSPOSE_COMM_NCCL_PL,
    TRANSPOSE_COMM_NVSHMEM,
    TRANSPOSE_COMM_NVSHMEM_PL,
    TRANSPOSE_XY,
    TRANSPOSE_YX,
    TRANSPOSE_YZ,
    TRANSPOSE_ZY,
    HaloCommBackend,
    TransposeCommBackend,
    finalize,
    get_autotuned_config,
    get_pencil_info,
    init,
    make_config,
)

try:
    __version__ = version('jaxDecomp')
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = [
    'config',
    'init',
    'finalize',
    'get_pencil_info',
    'get_autotuned_config',
    'make_config',
    'halo_exchange',
    'pfft3d',
    'pifft3d',
    'transposeXtoY',
    'transposeYtoX',
    'transposeYtoZ',
    'transposeZtoY',
    'transposeXtoZ',
    'transposeZtoX',
    'TRANSPOSE_XY',
    'TRANSPOSE_YX',
    'TRANSPOSE_YZ',
    'TRANSPOSE_ZY',
    'SLAB_XY',
    'SLAB_YZ',
    'PENCILS',
    'NO_DECOMP',
    'get_fft_output_sharding',
    'get_output_specs',
    'validate_spec_matches_mesh',
    'fftfreq3d',
    'rfftfreq3d',
    'HALO_COMM_MPI',
    'HALO_COMM_MPI_BLOCKING',
    'HALO_COMM_NVSHMEM',
    'HALO_COMM_NVSHMEM_BLOCKING',
    'TRANSPOSE_COMM_MPI_A2A',
    'TRANSPOSE_COMM_MPI_P2P',
    'TRANSPOSE_COMM_MPI_P2P_PL',
    'TRANSPOSE_COMM_NCCL_PL',
    'TRANSPOSE_COMM_NVSHMEM',
    'TRANSPOSE_COMM_NVSHMEM_PL',
]


@dataclass
class JAXDecompConfig:
    """Class for storing the configuration state of the library."""

    halo_comm_backend: HaloCommBackend = HALO_COMM_NCCL
    transpose_comm_backend: TransposeCommBackend = TRANSPOSE_COMM_NCCL
    transpose_axis_contiguous: bool = True

    def update(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise ValueError(f'key {key} is not a valid configuration key')


# Declare the global configuration object
config = JAXDecompConfig()
