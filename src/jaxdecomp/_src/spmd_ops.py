from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, Optional

import jax
import jax.extend as jex
from jax._src import custom_api_util, dispatch
from jax.experimental.custom_partitioning import custom_partitioning
from jax.interpreters import ad, batching, mlir, xla
from packaging.version import Version as _Version

if _Version(jax.__version__) >= _Version('0.7.0'):
    ALLOW_SHARDY_PARTITIONER = True
else:
    ALLOW_SHARDY_PARTITIONER = False


class BasePrimitive(metaclass=ABCMeta):
    """
    Base class for JAX primitives.
    """

    name: str
    multiple_results: bool
    impl_static_args: tuple[Any, ...]
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
    def sharding_rule_producer(*args, **kwargs) -> Any:
        """
        Produces sharding rule for custom partitioning with Shardy partitioner.

        This method should be implemented by subclasses to define how operations
        should be partitioned across devices when using JAX's Shardy partitioner.

        Parameters
        ----------
        *args
            Variable positional arguments specific to the operation.
        **kwargs
            Variable keyword arguments specific to the operation.

        Returns
        -------
        Any
            Sharding rule specification (typically an einsum string).

        Notes
        -----
        Base implementation returns NotImplemented and should be overridden
        by concrete primitive implementations.
        """
        return NotImplemented

    @staticmethod
    @abstractmethod
    def partition(*args, **kwargs) -> Any:
        """
        Describes the partitioning logic for custom partitioning.
        """
        return NotImplemented

    @staticmethod
    @abstractmethod
    def batching(*args, **kwargs) -> Any:
        """
        Describes the batching rule for batching.
        """
        return NotImplemented


def register_primitive(cls: type[BasePrimitive]) -> None:
    """
    Registers a JAX primitive.

    Args:
        cls: The primitive class to register a BasePrimitive

    Raises:
        ValueError: If the class is not a BasePrimitive
    """
    if issubclass(cls, BasePrimitive):

        def name_of_wrapper_p() -> str:
            return cls.name + '_wrapper'

        inner_p = jex.core.Primitive(cls.name)
        dispatch.prim_requires_devices_during_lowering.add(inner_p)
        inner_p.multiple_results = cls.multiple_results
        inner_p.def_impl(partial(xla.apply_primitive, inner_p))
        inner_p.def_abstract_eval(cls.abstract)
        mlir.register_lowering(inner_p, cls.lowering, platform='cuda')
        cls.inner_primitive = inner_p

        outer_p = jex.core.Primitive(name_of_wrapper_p())
        dispatch.prim_requires_devices_during_lowering.add(outer_p)
        outer_p.multiple_results = cls.multiple_results
        outer_p.def_impl(cls.impl)
        outer_p.def_abstract_eval(cls.outer_abstract)
        batching.primitive_batchers[outer_p] = cls.batching
        outer_p_lower = custom_partitioning(cls.impl, static_argnums=cls.impl_static_args)

        infer_sharding = None
        sharding_rule = None
        if jax.config.jax_use_shardy_partitioner and ALLOW_SHARDY_PARTITIONER:
            sharding_rule = cls.sharding_rule_producer
        else:
            infer_sharding = cls.infer_sharding_from_operands

        outer_p_lower.def_partition(
            infer_sharding_from_operands=infer_sharding,
            partition=cls.partition,
            sharding_rule=sharding_rule,
        )
        mlir.register_lowering(
            outer_p,
            mlir.lower_fun(outer_p_lower, multiple_results=cls.multiple_results),
        )
        cls.outer_primitive = outer_p
    else:
        raise ValueError('register_primitive only accepts BasePrimitive')


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
            return jax.make_jaxpr(self.fun, static_argnums=self.static_argnums)(*args, **kwargs).out_avals[0]

        self.primitive.def_abstract_eval(abstract_eval)

        # Functions to be registered
        self.partition = None
        self.infer_sharding_from_operands = None
        self.sharding_rule_producer = None
        self.jvp_rule = None
        self.transpose_rule = None
        self.batching_rule = None

    def def_partition(self, partition):
        self.partition = partition
        if self.sharding_rule_producer is not None and jax.config.jax_use_shardy_partitioner:
            self.def_spmd_rule(partition, self.infer_sharding_from_operands, self.sharding_rule_producer)
        if self.infer_sharding_from_operands is not None and not jax.config.jax_use_shardy_partitioner:
            self.def_spmd_rule(partition, self.infer_sharding_from_operands, self.sharding_rule_producer)

    def def_infer_sharding(self, infer_sharding_from_operands):
        self.infer_sharding_from_operands = infer_sharding_from_operands
        if self.partition is not None:
            self.def_spmd_rule(self.partition, infer_sharding_from_operands, self.sharding_rule_producer)

    def def_sharding_rule(self, sharding_rule_producer):
        self.sharding_rule_producer = sharding_rule_producer
        if self.partition is not None:
            self.def_spmd_rule(self.partition, self.infer_sharding_from_operands, sharding_rule_producer)

    def def_spmd_rule(self, partition_rule, infer_sharding_rule, sharding_rule_producer):
        assert partition_rule is not None, 'Partition rule is required'
        if jax.config.jax_use_shardy_partitioner:
            assert ALLOW_SHARDY_PARTITIONER, 'Shardy partitioner is not supported in this JAX version use at least JAX 0.7.0'

        if jax.config.jax_use_shardy_partitioner:
            assert sharding_rule_producer is not None, 'sharding_rule_producer is required when jax_use_shardy_partitioner is True'
        else:
            assert infer_sharding_rule is not None, 'infer_sharding_rule is required when jax_use_shardy_partitioner is False'

        partitioned_fn = custom_partitioning(self.fun, static_argnums=self.static_argnums)
        partitioned_fn.def_partition(
            infer_sharding_from_operands=infer_sharding_rule,
            partition=partition_rule,
            sharding_rule=sharding_rule_producer,
        )
        # ============== PRIMITIVE ==============
        #       Declare custom SPMD and batching rule
        # ======================================
        # Step 2: Register the Partitioned lowering and the batching rule
        mlir.register_lowering(
            self.primitive,
            mlir.lower_fun(partitioned_fn, multiple_results=self.multiple_results),
        )

    def def_jvp_rule(self, jvp_rule):
        self.jvp_rule = jvp_rule
        ad.primitive_jvps[self.primitive] = jvp_rule

    def def_transpose_rule(self, transpose_rule):
        self.transpose_rule = transpose_rule
        ad.primitive_transposes[self.primitive] = transpose_rule

    def def_batching_rule(self, batching_rule):
        self.batching_rule = batching_rule
        batching.primitive_batchers[self.primitive] = batching_rule

    def __call__(self, *args, **kwargs):
        def internal_call(*args, **kwargs):
            return self.primitive.bind(*args, **kwargs)

        return internal_call(*args, **kwargs)
