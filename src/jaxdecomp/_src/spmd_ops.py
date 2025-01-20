import sys
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, Hashable, Optional, Tuple, Type

import jax
from jax import core
from jax._src import dispatch
from jax._src import mesh as mesh_lib
from jax.experimental.custom_partitioning import custom_partitioning
from jax.interpreters import mlir, xla
from jax.sharding import Mesh, NamedSharding
from jaxdecomplib import _jaxdecomp

from jaxdecomp.typing import PdimsType, TransposablePdimsType

if sys.version_info < (3, 11):
    pass
else:
    pass


import jax.extend as jex
from jax._src import custom_api_util
from jax.interpreters import ad
# Imports

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
        if len(pdims) == 1:
            pdims = (1,) + pdims

        if len(pdims) != 2:
            raise ValueError("Only one or two-dimensional meshes are supported.")

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
    inner_primitive: Optional[jex.core.Primitive]
    outer_primitive: Optional[jex.core.Primitive]
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


def register_primitive(cls: Type[BasePrimitive]) -> None:
    """
    Registers a JAX primitive.

    Args:
        cls: The primitive class to register a BasePrimitive

    Raises:
        ValueError: If the class is not a BasePrimitive
    """
    if issubclass(cls, BasePrimitive):

        def name_of_wrapper_p() -> str:
            return cls.name + "_wrapper"

        inner_p = core.Primitive(cls.name)
        dispatch.prim_requires_devices_during_lowering.add(inner_p)
        inner_p.multiple_results = cls.multiple_results
        inner_p.def_impl(partial(xla.apply_primitive, inner_p))
        inner_p.def_abstract_eval(cls.abstract)
        mlir.register_lowering(inner_p, cls.lowering, platform="cuda")
        cls.inner_primitive = inner_p

        outer_p = core.Primitive(name_of_wrapper_p())
        dispatch.prim_requires_devices_during_lowering.add(outer_p)
        outer_p.multiple_results = cls.multiple_results
        outer_p.def_impl(cls.impl)
        outer_p.def_abstract_eval(cls.outer_abstract)
        outer_p_lower = custom_partitioning(
            cls.impl, static_argnums=cls.impl_static_args
        )
        outer_p_lower.def_partition(
            infer_sharding_from_operands=cls.infer_sharding_from_operands,
            partition=cls.partition,
        )
        mlir.register_lowering(
            outer_p,
            mlir.lower_fun(outer_p_lower, multiple_results=cls.multiple_results),
        )
        cls.outer_primitive = outer_p
    else:
        raise ValueError("register_primitive only accepts BasePrimitive")


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
    return tuple([get_axis_size(sharding, i) for i in range(len(sharding.spec))])  # type: ignore


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
        assert len(pdims) == 2

    return pdims


@custom_api_util.register_custom_decorator_type
class custom_spmd_rule:
    def __init__(self, fun, static_argnums=(), multiple_results=False):
        self.fun = fun
        self.static_argnums = static_argnums
        self.multiple_results = multiple_results

        # ============== PRIMITIVE ==============
        #       Declare primitive
        # ======================================
        self.primitive = jex.core.Primitive(fun.__name__)
        # This is needed for lowering custom spmd rule
        dispatch.prim_requires_devices_during_lowering.add(self.primitive)
        # Step 1: Define the Implementation and Abstract Evaluation
        self.primitive.def_impl(fun)

        def abstract_eval(*args, **kwargs):
            return jax.make_jaxpr(self.fun, static_argnums=self.static_argnums)(
                *args, **kwargs
            ).out_avals[0]

        self.primitive.def_abstract_eval(abstract_eval)

        # Functions to be registered
        self.partition = None
        self.infer_sharding_from_operands = None
        self.jvp_rule = None
        self.transpose_rule = None

    def def_partition(self, partition):
        self.partition = partition
        if self.infer_sharding_from_operands is not None:
            self.def_spmd_rule(partition, self.infer_sharding_from_operands)

    def def_infer_sharding(self, infer_sharding_from_operands):
        self.infer_sharding_from_operands = infer_sharding_from_operands
        if self.partition is not None:
            self.def_spmd_rule(self.partition, infer_sharding_from_operands)

    def def_spmd_rule(self, partition_rule, infer_sharding_rule):
        assert partition_rule is not None, "Partition rule is required"
        assert infer_sharding_rule is not None, "Infer sharding rule is required"

        paritioned_fn = custom_partitioning(
            self.fun, static_argnums=self.static_argnums
        )
        paritioned_fn.def_partition(
            infer_sharding_from_operands=infer_sharding_rule,
            partition=partition_rule,
        )
        # ============== PRIMITIVE ==============
        #       Declare custom SPMD and batching rule
        # ======================================
        # Step 2: Register the Partitioned lowering and the batching rule
        mlir.register_lowering(
            self.primitive,
            mlir.lower_fun(paritioned_fn, multiple_results=self.multiple_results),
        )

    def def_jvp_rule(self, jvp_rule):
        self.jvp_rule = jvp_rule
        ad.primitive_jvps[self.primitive] = jvp_rule

    def def_transpose_rule(self, transpose_rule):
        self.transpose_rule = transpose_rule
        ad.primitive_transposes[self.primitive] = transpose_rule

    def __call__(self, *args, **kwargs):
        def internal_call(*args, **kwargs):
            return self.primitive.bind(*args, **kwargs)

        return internal_call(*args, **kwargs)
