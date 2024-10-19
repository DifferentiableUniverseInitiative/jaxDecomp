from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Tuple

from jaxdecomp._src.pencil_utils import get_output_specs
from jaxdecomp.fft import fftfreq3d_shard, pfft3d, pifft3d
from jaxdecomp.halo import halo_exchange
from jaxdecomp.transpose import (transposeXtoY, transposeXtoZ, transposeYtoX,
                                 transposeYtoZ, transposeZtoX, transposeZtoY)

from ._src import (HALO_COMM_MPI, HALO_COMM_MPI_BLOCKING, HALO_COMM_NCCL,
                   HALO_COMM_NVSHMEM, HALO_COMM_NVSHMEM_BLOCKING, NO_DECOMP,
                   PENCILS, SLAB_XY, SLAB_YZ, TRANSPOSE_COMM_MPI_A2A,
                   TRANSPOSE_COMM_MPI_P2P, TRANSPOSE_COMM_MPI_P2P_PL,
                   TRANSPOSE_COMM_NCCL, TRANSPOSE_COMM_NCCL_PL,
                   TRANSPOSE_COMM_NVSHMEM, TRANSPOSE_COMM_NVSHMEM_PL,
                   TRANSPOSE_XY, TRANSPOSE_YX, TRANSPOSE_YZ, TRANSPOSE_ZY,
                   HaloCommBackend, TransposeCommBackend, finalize,
                   get_autotuned_config, get_pencil_info, init, make_config,
                   slice_pad, slice_unpad)

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
    "get_autotuned_config",
    "make_config",
    "halo_exchange",
    "slice_pad",
    "slice_unpad",
    "pfft3d",
    "pifft3d",
    "transposeXtoY",
    "transposeYtoX",
    "transposeYtoZ",
    "transposeZtoY",
    "transposeXtoZ",
    "transposeZtoX",
    "TRANSPOSE_XY",
    "TRANSPOSE_YX",
    "TRANSPOSE_YZ",
    "TRANSPOSE_ZY",
    "SLAB_XY",
    "SLAB_YZ",
    "PENCILS",
    "NO_DECOMP",
    "get_output_specs",
    "fftfreq3d_shard",
]


@dataclass
class JAXDecompConfig:
  """Class for storing the configuration state of the library."""
  halo_comm_backend: HaloCommBackend = HALO_COMM_NCCL
  transpose_comm_backend: TransposeCommBackend = TRANSPOSE_COMM_NCCL
  transpose_axis_contiguous: bool = True
  transpose_axis_contiguous_2: bool = True

  def update(self, key, value):
    if hasattr(self, key):
      setattr(self, key, value)
    else:
      raise ValueError("key %s is not a valid configuration key" % key)


# Declare the global configuration object
config = JAXDecompConfig()
