from jaxtyping import Array

from jaxdecomp._src.cudecomp.halo import \
    halo_exchange as _cudecomp_halo_exchange
from jaxdecomp._src.jax.halo import HaloExtentType, Periodicity
from jaxdecomp._src.jax.halo import halo_exchange as _jax_halo_exchange


def halo_exchange(
    x: Array,
    halo_extents: HaloExtentType,
    halo_periods: Periodicity,
    backend: str = "jax",
) -> Array:
  """
    Perform a halo exchange operation using the specified backend.

    Parameters
    ----------
    x : Array
        Input array for the halo exchange.
    halo_extents : HaloExtentType
        Tuple specifying the extents of the halo in each dimension.
    halo_periods : Periodicity
        Tuple specifying the periodicity (True or False) in each dimension.
    backend : str, optional
        Backend to use for the halo exchange ("jax" or "cudecomp"), by default "jax".

    Returns
    -------
    Array
        Array after performing the halo exchange.

    Raises
    ------
    ValueError
        If an invalid backend is specified.

    Example
    -------
    >>> import jax
    >>> from jax import random
    >>> from jax.sharding import PartitionSpec as P
    >>> from jax.experimental import mesh_utils
    >>> from jax.sharding import Mesh, NamedSharding
    >>> from jax.experimental.shard_map import shard_map

    # Initialize distributed mesh and array
    >>> jax.distributed.initialize()
    >>> rank = jax.process_index()
    >>> pdims = (2, 2)
    >>> global_shape = (16, 16, 16)
    >>> local_shape = (global_shape[0] // pdims[0], global_shape[1] // pdims[1], global_shape[2])
    >>> devices = mesh_utils.create_device_mesh(pdims)
    >>> mesh = Mesh(devices.T, axis_names=('z', 'y'))
    >>> sharding = NamedSharding(mesh, P('z', 'y'))

    # Create global array with random values
    >>> global_array = jax.make_array_from_callback(global_shape, sharding, lambda idx: random.normal(random.PRNGKey(rank), local_shape))

    # Define padding for halo exchange
    >>> padding = [(1, 1), (1, 1), (0, 0)]

    # Pad the array with custom padding
    >>> @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
    >>> def pad(arr):
    ...     return jnp.pad(arr, padding, mode='linear_ramp', end_values=20)

    >>> padded_array = pad(global_array)

    # Perform halo exchange using JAX backend
    >>> halo_extents = (1, 1)
    >>> halo_periods = (True, True)
    >>> updated_array = halo_exchange(padded_array, halo_extents, halo_periods, backend="jax")
    """
  if backend.lower() == "jax":
    return _jax_halo_exchange(x, halo_extents, halo_periods)
  elif backend.lower() == "cudecomp":
    return _cudecomp_halo_exchange(x, halo_extents, halo_periods)
  else:
    raise ValueError(f"Invalid backend: {backend}")
