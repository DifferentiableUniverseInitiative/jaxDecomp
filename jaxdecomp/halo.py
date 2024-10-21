from typing import Tuple

from jaxtyping import Array

from jaxdecomp._src.cudecomp.halo import \
    halo_exchange as _cudecomp_halo_exchange
from jaxdecomp._src.jax.halo import HaloExtentType, Periodicity
from jaxdecomp._src.jax.halo import halo_exchange as _jax_halo_exchange


def halo_exchange(x: Array,
                  halo_extents: HaloExtentType,
                  halo_periods: Periodicity,
                  backend: str = "jax") -> Array:
  if backend.lower() == "jax":
    return _jax_halo_exchange(x, halo_extents, halo_periods)
  elif backend.lower() == "cudecomp":
    return _cudecomp_halo_exchange(x, halo_extents, halo_periods)
  else:
    raise ValueError(f"Invalid backend: {backend}")
