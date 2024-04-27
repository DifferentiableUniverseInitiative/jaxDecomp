from dataclasses import dataclass
from ._src import finalize, get_pencil_info, get_autotuned_config, make_config, halo_exchange, slice_pad, slice_unpad #transposeXtoY, transposeYtoX, transposeYtoZ, transposeZtoY
from ._src import HALO_COMM_MPI, HALO_COMM_MPI_BLOCKING, HALO_COMM_NCCL, HALO_COMM_NVSHMEM, HALO_COMM_NVSHMEM_BLOCKING
from ._src import TRANSPOSE_COMM_MPI_A2A, TRANSPOSE_COMM_MPI_P2P, TRANSPOSE_COMM_MPI_P2P_PL, TRANSPOSE_COMM_NCCL, TRANSPOSE_COMM_NCCL_PL, TRANSPOSE_COMM_NVSHMEM, TRANSPOSE_COMM_NVSHMEM_PL
from ._src import HaloCommBackend, TransposeCommBackend
import jaxdecomp.fft as fft
from jaxdecomp.fft import pfft3d, pifft3d

from importlib.metadata import version, PackageNotFoundError

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
