import operator
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Hashable, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from collections.abc import Callable


# Imports

Specs = Any
AxisName = Hashable


@jax.tree_util.register_dataclass
@dataclass
class ShardedArray:
    data: jax.Array
    initial_sharding: NamedSharding | None = field(
        default=None, metadata=dict(static=True)
    )

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def reshape(self, *shape: int) -> Self:
        return jax.tree.map(lambda x: x.reshape(*shape), self)

    def astype(self, dtype: Any) -> Self:
        return jax.tree.map(lambda x: x.astype(dtype), self)

    def prod(self, *args, **kwargs) -> Any:
        return jax.tree.map(lambda x: x.prod(*args, **kwargs), self)

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    @property
    def real(self) -> Self:
        return jax.tree.map(lambda x: x.real, self)

    @property
    def imag(self) -> Self:
        return jax.tree.map(lambda x: x.imag, self)

    @property
    def sharding(self) -> Any:
        return self.data.sharding

    def transpose(self, *args, **kwargs) -> Self:
        return jax.tree.map(lambda x: x.transpose(*args, **kwargs), self)

    def T(self) -> Self:
        return jax.tree.map(lambda x: x.T, self)

    def mean(self, *args, **kwargs) -> Self:
        return jax.tree.map(lambda x: x.mean(*args, **kwargs), self)

    def sum(self, *args, **kwargs) -> Self:
        return jax.tree.map(lambda x: x.sum(*args, **kwargs), self)

    def __repr__(self) -> str:
        return f"ShardedArray(data={self.data}, sharding={self.initial_sharding})"

    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, ShardedArray):
            return ShardedArray(self.data @ other.data, self.initial_sharding)
        elif isinstance(other, jax.numpy.ndarray):
            return ShardedArray(self.data @ other, self.initial_sharding)
        else:
            return NotImplemented

    def __len__(self) -> int:
        return len(self.data)

    def __abs__(self) -> Self:
        result: Self = jax.tree.map(operator.abs, self)
        return result

    def __pos__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        result: Self = jax.tree.map(operator.neg, self)
        return result

    def __add__(self, other: Any) -> Self:
        return self._operation(operator.add, other)

    def __mod__(self, other: Any) -> Self:
        return self._operation(operator.mod, other)

    def __sub__(self, other: Any) -> Self:
        return self._operation(operator.sub, other)

    def __mul__(self, other: Any) -> Self:
        return self._operation(operator.mul, other)

    def __truediv__(self, other: Any) -> Self:
        return self._operation(operator.truediv, other)

    def __pow__(self, other: Any) -> Self:
        return self._operation(operator.pow, other)

    def __eq__(self, other: Any) -> Any:
        if isinstance(other, type(self)):
            return self.data == other.data
        elif jnp.isscalar(other) or isinstance(other, jax.Array):
            return self.data == other

    def __le__(self, other: Any) -> Any:
        if isinstance(other, type(self)):
            return self.data <= other.data
        elif jnp.isscalar(other) or isinstance(other, jax.Array):
            return self.data <= other
        else:
            return NotImplemented

    def __bool__(self) -> bool:
        return bool(self.data)

    def __lt__(self, other: Any) -> Any:
        if isinstance(other, type(self)):
            return self.data < other.data
        elif jnp.isscalar(other) or isinstance(other, jax.Array):
            return self.data < other
        else:
            return NotImplemented

    def __ge__(self, other: Any) -> Any:
        if isinstance(other, type(self)):
            return self.data >= other.data
        elif jnp.isscalar(other) or isinstance(other, jax.Array):
            return self.data >= other
        else:
            return NotImplemented

    def __gt__(self, other: Any) -> Any:
        if isinstance(other, type(self)):
            return self.data > other.data
        elif jnp.isscalar(other) or isinstance(other, jax.Array):
            return self.data > other
        else:
            return NotImplemented

    def __iter__(self) -> Any:
        if self.ndim == 0:
            raise TypeError(f"'{type(self).__name__}' object is not iterable")
        return iter([jax.tree.map(lambda x: x[i], self) for i in range(self.shape[0])])

    def nonzero(self) -> Any:
        return jax.tree.map(lambda x: x.nonzero(), self)

    @property
    def at(self) -> Any:
        return jax.tree.map(lambda x: x.at, self)

    def set(self, value: Any) -> Any:
        if isinstance(value, type(self)):
            return jax.tree.map(lambda x, y: x.set(y), self, value)
        elif jnp.isscalar(value) or isinstance(value, jax.Array):
            return jax.tree.map(lambda x: x.set(value), self)

    def add(self, value: Any) -> Any:
        if isinstance(value, type(self)):
            return jax.tree.map(lambda x, y: x.add(y), self, value)
        elif jnp.isscalar(value) or isinstance(value, jax.Array):
            return jax.tree.map(lambda x: x.add(value), self)

    def substract(self, value: Any) -> Any:
        if isinstance(value, type(self)):
            return jax.tree.map(lambda x, y: x.substract(y), self, value)
        elif jnp.isscalar(value) or isinstance(value, jax.Array):
            return jax.tree.map(lambda x: x.substract(value), self)

    def multiply(self, value: Any) -> Any:
        if isinstance(value, type(self)):
            return jax.tree.map(lambda x, y: x.multiply(y), self, value)
        elif jnp.isscalar(value) or isinstance(value, jax.Array):
            return jax.tree.map(lambda x: x.multiply(value), self)

    def divide(self, value: Any) -> Any:
        if isinstance(value, type(self)):
            return jax.tree.map(lambda x, y: x.divide(y), self, value)
        elif jnp.isscalar(value) or isinstance(value, jax.Array):
            return jax.tree.map(lambda x: x.divide(value), self)

    def power(self, value: Any) -> Any:
        if isinstance(value, type(self)):
            return jax.tree.map(lambda x, y: x.power(y), self, value)
        elif jnp.isscalar(value) or isinstance(value, jax.Array):
            return jax.tree.map(lambda x: x.power(value), self)

    def min(self, value: Any) -> Any:
        if isinstance(value, type(self)):
            return jax.tree.map(lambda x, y: x.min(y), self, value)
        elif jnp.isscalar(value) or isinstance(value, jax.Array):
            return jax.tree.map(lambda x: x.min(value), self)

    def max(self, value: Any) -> Any:
        if isinstance(value, type(self)):
            return jax.tree.map(lambda x, y: x.max(y), self, value)
        elif jnp.isscalar(value) or isinstance(value, jax.Array):
            return jax.tree.map(lambda x: x.max(value), self)

    def apply(self, func: Callable[[Any], Any]) -> Any:
        return jax.tree.map(lambda x: x.apply(func), self)

    def get(self, *args, **kwargs) -> Any:
        return jax.tree.map(lambda x: x.get(*args, **kwargs), self)

    def _operation(self, operation: Callable[[Any, Any], Any], right: Any) -> Self:
        result: Self
        if isinstance(right, type(self)):
            result = jax.tree.map(operation, self, right)
        elif jnp.isscalar(right) or isinstance(right, jax.Array):
            result = jax.tree.map(lambda leaf: operation(leaf, right), self)
        else:
            return NotImplemented
        return result

    def __rmatmul__(self, other: Any) -> Any:
        if isinstance(other, ShardedArray):
            return ShardedArray(other.data @ self.data, self.initial_sharding)
        elif isinstance(other, jax.numpy.ndarray):
            return ShardedArray(other @ self.data, self.initial_sharding)
        else:
            return NotImplemented

    def __getitem__(self, key: Any) -> Any:
        return jax.tree.map(lambda x: x[key], self)

    def __radd__(self, other: Any) -> Self:
        return self._roperation(operator.add, other)

    def __floordiv__(self, other: Any) -> Self:
        return self._operation(operator.floordiv, other)

    def __rfloordiv__(self, other: Any) -> Self:
        return self._roperation(operator.floordiv, other)

    def __rmod(self, other: Any) -> Self:
        return self._roperation(operator.mod, other)

    def __rsub__(self, other: Any) -> Self:
        return self._roperation(operator.sub, other)

    def __rmul__(self, other: Any) -> Self:
        return self._roperation(operator.mul, other)

    def __rtruediv__(self, other: Any) -> Self:
        return self._roperation(operator.truediv, other)

    def __rpow__(self, other: Any) -> Self:
        return self._roperation(operator.pow, other)

    def __req__(self, other: Any) -> Any:
        if isinstance(other, type(self)):
            return other.data == self.data
        elif jnp.isscalar(other) or isinstance(other, jax.Array):
            return other == self.data

    def _roperation(self, operation: Callable[[Any, Any], Any], left: Any) -> Self:
        result: Self
        if isinstance(left, type(self)):
            result = jax.tree.map(operation, left, self)
        elif jnp.isscalar(left) or isinstance(left, jax.Array):
            result = jax.tree.map(partial(operation, left), self)
        else:
            return NotImplemented
        return result
