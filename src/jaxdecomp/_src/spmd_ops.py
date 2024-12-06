from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, Hashable, Optional, Tuple, Type, Union

from jax import core
from jax._src import dispatch
from jax._src import mesh as mesh_lib
from jax._src.interpreters import batching
from jax.experimental.custom_partitioning import custom_partitioning
from jax.interpreters import mlir, xla
from jax.sharding import Mesh, NamedSharding
from jaxdecomplib import _jaxdecomp

from jaxdecomp.typing import PdimsType, TransposablePdimsType

Specs = Any
AxisName = Hashable


def get_pencil_type(mesh: Optional[Mesh] = None) -> str:
  """Returns the pencil decomposition type based on the mesh configuration.

    Args:
        mesh: The device mesh (Optional[Mesh]). If not provided, uses the current physical mesh.

    Returns:
        A string representing the pencil decomposition type.

    Raises:
        ValueError: If an unknown pencil type is encountered.
    """
  if mesh is None:
    mesh = mesh_lib.thread_resources.env.physical_mesh
  if mesh.empty:
    pdims = None
  else:
    pdims = mesh.devices.shape[::-1]

  if pdims == (1, 1) or pdims is None:
    return _jaxdecomp.NO_DECOMP
  elif pdims[0] == 1:
    return _jaxdecomp.SLAB_XY
  elif pdims[1] == 1:
    return _jaxdecomp.SLAB_YZ
  else:
    return _jaxdecomp.PENCILS


class BasePrimitive(metaclass=ABCMeta):
  """
    Base class for JAX primitives.
    """
  name: str
  multiple_results: bool
  impl_static_args: Tuple[Any, ...]
  inner_primitive: Optional[core.Primitive]
  outer_primitive: Optional[core.Primitive]
  outer_lowering: custom_partitioning

  @staticmethod
  @abstractmethod
  def abstract(*args, **kwargs) -> Any:
    """
        Describes the abstract evaluation of the primitive in the JAX computation graph.
        """
    return NotImplemented

  @classmethod
  def outer_abstract(cls, *args, **kwargs) -> Any:
    """
        Optional abstract wrapper to eliminate workspace tensors.
        """
    return cls.abstract(*args, **kwargs)

  @staticmethod
  @abstractmethod
  def lowering(*args, **kwargs) -> Any:
    """
        Describes the MLIR lowering of the primitive.
        """
    return NotImplemented

  @staticmethod
  @abstractmethod
  def impl(*args, **kwargs) -> Any:
    """
        Describes the implementation of the primitive.
        """
    return NotImplemented

  @staticmethod
  @abstractmethod
  def per_shard_impl(*args, **kwargs) -> Any:
    """
        Describes the per-shard implementation for custom partitioning.
        """
    return NotImplemented

  @staticmethod
  @abstractmethod
  def batcher(*args, **kwargs) -> Any:
    """
        Describes the batch rules for vmap.
        """
    return NotImplemented

  @staticmethod
  @abstractmethod
  def infer_sharding_from_operands(*args, **kwargs) -> Any:
    """
        Infers sharding from the operands for custom partitioning.
        """
    return NotImplemented

  @staticmethod
  @abstractmethod
  def partition(*args, **kwargs) -> Any:
    """
        Describes the partitioning logic for custom partitioning.
        """
    return NotImplemented


class CustomParPrimitive(metaclass=ABCMeta):
  """
    SPMD Custom Partitioning wrapper.
    """
  name: str
  multiple_results: bool
  impl_static_args: Tuple[Any, ...]
  outer_lowering: custom_partitioning

  @staticmethod
  @abstractmethod
  def impl(*args, **kwargs) -> Any:
    """
        Describes the implementation of the primitive.
        """
    return NotImplemented

  @staticmethod
  @abstractmethod
  def per_shard_impl(*args, **kwargs) -> Any:
    """
        Describes the per-shard implementation for custom partitioning.
        """
    return NotImplemented

  @staticmethod
  @abstractmethod
  def infer_sharding_from_operands(*args, **kwargs) -> Any:
    """
        Infers sharding from the operands for custom partitioning.
        """
    return NotImplemented

  @staticmethod
  @abstractmethod
  def partition(*args, **kwargs) -> Any:
    """
        Describes the partitioning logic for custom partitioning.
        """
    return NotImplemented


def register_primitive(
    cls: Type[Union[BasePrimitive, CustomParPrimitive]]) -> None:
  """
    Registers a JAX primitive.

    Args:
        cls: The primitive class to register, either a BasePrimitive or CustomParPrimitive.

    Raises:
        ValueError: If the class is not a BasePrimitive or CustomParPrimitive.
    """
  if issubclass(cls, BasePrimitive):

    def name_of_wrapper_p() -> str:
      return cls.name + "_wrapper"

    inner_p = core.Primitive(cls.name)
    dispatch.prim_requires_devices_during_lowering.add(inner_p)
    inner_p.multiple_results = cls.multiple_results
    inner_p.def_impl(partial(xla.apply_primitive, inner_p))
    inner_p.def_abstract_eval(cls.abstract)
    mlir.register_lowering(inner_p, cls.lowering, platform='cuda')
    cls.inner_primitive = inner_p

    outer_p = core.Primitive(name_of_wrapper_p())
    dispatch.prim_requires_devices_during_lowering.add(outer_p)
    outer_p.multiple_results = cls.multiple_results
    outer_p.def_impl(cls.impl)
    outer_p.def_abstract_eval(cls.outer_abstract)
    batching.primitive_batchers[outer_p] = cls.batcher
    outer_p_lower = custom_partitioning(
        cls.impl, static_argnums=cls.impl_static_args)
    outer_p_lower.def_partition(
        infer_sharding_from_operands=cls.infer_sharding_from_operands,
        partition=cls.partition)
    mlir.register_lowering(
        outer_p,
        mlir.lower_fun(outer_p_lower, multiple_results=cls.multiple_results))
    cls.outer_primitive = outer_p

  elif issubclass(cls, CustomParPrimitive):
    outer_p_lower = custom_partitioning(
        cls.impl, static_argnums=cls.impl_static_args)
    outer_p_lower.def_partition(
        infer_sharding_from_operands=cls.infer_sharding_from_operands,
        partition=cls.partition)
    cls.outer_lowering = outer_p_lower

  else:
    raise ValueError(
        "register_primitive only accepts BasePrimitive or CustomParPrimitive")


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
  return tuple([get_axis_size(sharding, i) for i in range(len(sharding.spec))
               ])  # type: ignore


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
    assert (len(pdims) == 2)

  return pdims
