from typing import Tuple

from jaxtyping import Array

from jaxdecomp._src.cudecomp.halo import \
    halo_exchange as _cudecomp_halo_exchange
from jaxdecomp._src.jax.halo import halo_exchange as _jax_halo_exchange


def halo_exchange(x: Array,
                  halo_extent: int,
                  periodic: bool,
                  backend: str = "jax") -> Array:
  if backend.lower() == "jax":
    return _jax_halo_exchange(x, halo_extent, periodic)
  elif backend.lower() == "cudecomp":
    return _cudecomp_halo_exchange(x, halo_extent, periodic)
  else:
    raise ValueError(f"Invalid backend: {backend}")
