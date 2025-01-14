from jaxtyping import Array

from jaxdecomp._src.cudecomp.transpose import transpose as _cudecomp_transpose
from jaxdecomp._src.jax.transpose import transpose as _jax_transpose


def transposeXtoY(x: Array, backend: str = "jax") -> Array:
    """
    Transpose the input array from X-pencil to Y-pencil.

    Note:
    Expects input in Z Y X format and returns output in X Z Y format.

    Parameters
    ----------
    x : Array
        Input array in Z Y X format to transpose.
    backend : str, optional
        Backend to use for the transpose operation ("jax" or "cudecomp"), by default "jax".

    Returns
    -------
    Array
        Transposed array in X Z Y format.

    Raises
    ------
    ValueError
        If the backend is invalid.

    Example
    -------
    >>> import jax
    >>> from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    >>> from jax.experimental import mesh_utils
    >>> import jaxdecomp
    >>> global_shape = (16, 32, 64)
    >>> pdims = (2, 4)
    >>> devices = mesh_utils.create_device_mesh(pdims)
    >>> mesh = Mesh(devices.T, axis_names=('z', 'y'))
    >>> sharding = NamedSharding(mesh, P('z', 'y'))
    >>> local_shape = (global_shape[0] // pdims[1], global_shape[1] // pdims[0], global_shape[2])
    >>> global_array = jax.make_array_from_callback(global_shape, sharding, data_callback=lambda _: jax.random.normal(jax.random.PRNGKey(0), local_shape))
    >>> transposed_array = jaxdecomp.transposeXtoY(global_array)
    """
    if backend.lower() == "jax":
        return _jax_transpose(x, kind="x_y")
    elif backend.lower() == "cudecomp":
        return _cudecomp_transpose(x, kind="x_y")
    else:
        raise ValueError(f"Invalid backend: {backend}")


def transposeYtoX(x: Array, backend: str = "jax") -> Array:
    """
    Transpose the input array from Y-pencil to X-pencil.

    Note:
    Expects input in X Z Y format and returns output in Z Y X format.

    Parameters
    ----------
    x : Array
        Input array in X Z Y format to transpose.
    backend : str, optional
        Backend to use for the transpose operation ("jax" or "cudecomp"), by default "jax".

    Returns
    -------
    Array
        Transposed array in Z Y X format.

    Raises
    ------
    ValueError
        If the backend is invalid.

    Example
    -------
    >>> import jaxdecomp
    >>> transposed_array = jaxdecomp.transposeYtoX(global_array)
    """
    if backend.lower() == "jax":
        return _jax_transpose(x, kind="y_x")
    elif backend.lower() == "cudecomp":
        return _cudecomp_transpose(x, kind="y_x")
    else:
        raise ValueError(f"Invalid backend: {backend}")


def transposeYtoZ(x: Array, backend: str = "jax") -> Array:
    """
    Transpose the input array from Y-pencil to Z-pencil.

    Note:
    Expects input in X Z Y format and returns output in Y X Z format.

    Parameters
    ----------
    x : Array
        Input array in X Z Y format to transpose.
    backend : str, optional
        Backend to use for the transpose operation ("jax" or "cudecomp"), by default "jax".

    Returns
    -------
    Array
        Transposed array in Y X Z format.

    Raises
    ------
    ValueError
        If the backend is invalid.

    Example
    -------
    >>> import jaxdecomp
    >>> transposed_array = jaxdecomp.transposeYtoZ(global_array)
    """
    if backend.lower() == "jax":
        return _jax_transpose(x, kind="y_z")
    elif backend.lower() == "cudecomp":
        return _cudecomp_transpose(x, kind="y_z")
    else:
        raise ValueError(f"Invalid backend: {backend}")


def transposeZtoY(x: Array, backend: str = "jax") -> Array:
    """
    Transpose the input array from Z-pencil to Y-pencil.

    Note:
    Expects input in Y X Z format and returns output in X Z Y format.

    Parameters
    ----------
    x : Array
        Input array in Y X Z format to transpose.
    backend : str, optional
        Backend to use for the transpose operation ("jax" or "cudecomp"), by default "jax".

    Returns
    -------
    Array
        Transposed array in X Z Y format.

    Raises
    ------
    ValueError
        If the backend is invalid.

    Example
    -------
    >>> import jaxdecomp
    >>> transposed_array = jaxdecomp.transposeZtoY(global_array)
    """
    if backend.lower() == "jax":
        return _jax_transpose(x, kind="z_y")
    elif backend.lower() == "cudecomp":
        return _cudecomp_transpose(x, kind="z_y")
    else:
        raise ValueError(f"Invalid backend: {backend}")


def transposeXtoZ(x: Array, backend: str = "jax") -> Array:
    """
    Transpose the input array from X-pencil to Z-pencil.

    Note:
    Expects input in Z Y X format and returns output in Z X Y format.

    Parameters
    ----------
    x : Array
        Input array in Z Y X format to transpose.
    backend : str, optional
        Backend to use for the transpose operation, by default "jax".

    Returns
    -------
    Array
        Transposed array in Z X Y format.

    Raises
    ------
    ValueError
        If the backend is invalid.
    NotImplementedError
        If the backend does not support the operation (e.g., 'cudecomp' for x_z).

    Example
    -------
    >>> import jaxdecomp
    >>> transposed_array = jaxdecomp.transposeXtoZ(global_array)
    """
    if backend.lower() == "jax":
        return _jax_transpose(x, kind="x_z")
    elif backend.lower() == "cudecomp":
        raise NotImplementedError("Cudecomp does not support x_z transpose")
    else:
        raise ValueError(f"Invalid backend: {backend}")


def transposeZtoX(x: Array, backend: str = "jax") -> Array:
    """
    Transpose the input array from Z-pencil to X-pencil.

    Note:
    Expects input in Y Z X format and returns output in X Z Y format.

    Parameters
    ----------
    x : Array
        Input array in Y Z X format to transpose.
    backend : str, optional
        Backend to use for the transpose operation, by default "jax".

    Returns
    -------
    Array
        Transposed array in X Z Y format.

    Raises
    ------
    ValueError
        If the backend is invalid.
    NotImplementedError
        If the backend does not support the operation (e.g., 'cudecomp' for z_x).

    Example
    -------
    >>> import jaxdecomp
    >>> transposed_array = jaxdecomp.transposeZtoX(global_array)
    """
    if backend.lower() == "jax":
        return _jax_transpose(x, kind="z_x")
    elif backend.lower() == "cudecomp":
        raise NotImplementedError("Cudecomp does not support z_x transpose")
    else:
        raise ValueError(f"Invalid backend: {backend}")
