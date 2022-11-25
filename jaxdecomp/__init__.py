from dataclasses import dataclass
from ._src import init, finalize, get_pencil_info, make_config, transposeXtoY, transposeYtoX, transposeYtoZ, transposeZtoY, halo_exchange
from ._src import _jaxdecomp
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
    "init",
    "finalize",
    "get_pencil_info",
    "make_config",
    "halo_exchange",
    "pfft3d",
    "pifft3d",
    "transposeXtoY",
    "transposeYtoZ",
    "transposeZtoY",
    "transposeYtoX",
]

# Loading the comm configuration flags at the top level
from ._src._jaxdecomp import HALO_COMM_MPI, HALO_COMM_MPI_BLOCKING, HALO_COMM_NCCL, HALO_COMM_NVSHMEM, HALO_COMM_NVSHMEM_BLOCKING
from ._src._jaxdecomp import TRANSPOSE_COMM_MPI_A2A, TRANSPOSE_COMM_MPI_P2P, TRANSPOSE_COMM_MPI_P2P_PL, TRANSPOSE_COMM_NCCL, TRANSPOSE_COMM_NCCL_PL, TRANSPOSE_COMM_NVSHMEM, TRANSPOSE_COMM_NVSHMEM_PL


@dataclass
class JAXDecompConfig:
  """Class for storing the configuration state of the library."""
  halo_comm_backend: _jaxdecomp.HaloCommBackend = HALO_COMM_NCCL
  transpose_comm_backend: _jaxdecomp.TransposeCommBackend = TRANSPOSE_COMM_NCCL

  def update(self, key, value):
    if hasattr(self, key):
      setattr(self, key, value)
    else:
      raise ValueError("key %s is not a valid configuration key" % key)


# Declare the global configuration object
config = JAXDecompConfig()
