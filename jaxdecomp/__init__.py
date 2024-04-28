from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

import jaxdecomp.fft as fft
from jaxdecomp.fft import pfft3d, pifft3d

from ._src import (  # transposeXtoY, transposeYtoX, transposeYtoZ, transposeZtoY
    HALO_COMM_MPI, HALO_COMM_MPI_BLOCKING, HALO_COMM_NCCL, HALO_COMM_NVSHMEM,
    HALO_COMM_NVSHMEM_BLOCKING, TRANSPOSE_COMM_MPI_A2A, TRANSPOSE_COMM_MPI_P2P,
    TRANSPOSE_COMM_MPI_P2P_PL, TRANSPOSE_COMM_NCCL, TRANSPOSE_COMM_NCCL_PL,
    TRANSPOSE_COMM_NVSHMEM, TRANSPOSE_COMM_NVSHMEM_PL, HaloCommBackend,
    TransposeCommBackend, finalize, get_autotuned_config, get_pencil_info,
    halo_exchange, make_config, slice_pad, slice_unpad)

try:
  __version__ = version("jaxDecomp")
except PackageNotFoundError:
  # package is not installed
  pass

__all__ = [
    "config",
    "finalize",
    "get_pencil_info",
    "get_autotuned_config",
    "make_config",
    "halo_exchange",
    "slice_pad",
    "slice_unpad",
    "pfft3d",
    "pifft3d",
    # Transpose functions are still in development
    # "transposeXtoY",
    # "transposeYtoZ",
    # "transposeZtoY",
    # "transposeYtoX",
]


@dataclass
class JAXDecompConfig:
  """Class for storing the configuration state of the library."""
  halo_comm_backend: HaloCommBackend = HALO_COMM_NCCL
  transpose_comm_backend: TransposeCommBackend = TRANSPOSE_COMM_NCCL

  def update(self, key, value):
    if hasattr(self, key):
      setattr(self, key, value)
    else:
      raise ValueError("key %s is not a valid configuration key" % key)


# Declare the global configuration object
config = JAXDecompConfig()
