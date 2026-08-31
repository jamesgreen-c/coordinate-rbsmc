from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.scipy.linalg import solve_triangular

from rbsmc.utils.mvn import mvn_logpdf


class DistParam(ABC):
    @property
    @abstractmethod
    def nat_param(self) -> NatParam:
        pass

    @abstractmethod
    def sample(self, key: Array, shape: Sequence[int] = ()) -> Array:
        pass

    @abstractmethod
    def log_pdf(self, value: Array):
        pass


class NatParam(ABC):
    @abstractmethod
    def _fields(self) -> Mapping[str, Array]:
        pass

    @classmethod
    @abstractmethod
    def _from_fields(cls, fields: Mapping[str, Array]) -> NatParam:
        pass

    def __add__(self, other: NatParam) -> NatParam:
        if type(self) is not type(other):
            raise TypeError("Natural parameters must have the same type.")
        left, right = self._fields(), other._fields()
        return type(self)._from_fields({name: left[name] + right[name] for name in left})

    def sum(self, axis: int | tuple[int, ...]) -> NatParam:
        return type(self)._from_fields({name: value.sum(axis=axis) for name, value in self._fields().items()})

    @property
    @abstractmethod
    def dist_param(self) -> DistParam:
        pass


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GaussianNatParam(NatParam):
    precision: Array
    precision_mean: Array

    def _fields(self) -> Mapping[str, Array]:
        return {"precision": self.precision, "precision_mean": self.precision_mean}

    @classmethod
    def _from_fields(cls, fields: Mapping[str, Array]) -> GaussianNatParam:
        return cls(fields["precision"], fields["precision_mean"])

    @property
    def dist_param(self) -> GaussianDistParam:
        chol = jnp.linalg.cholesky(self.precision)
        eye = jnp.eye(self.precision.shape[-1], dtype=self.precision.dtype)
        cov = solve_triangular(chol.T, solve_triangular(chol, eye, lower=True), lower=False)
        mean = cov @ self.precision_mean
        return GaussianDistParam(mean, 0.5 * (cov + cov.T))

    def tree_flatten(self):
        return (self.precision, self.precision_mean), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GaussianDistParam(DistParam):
    mean: Array
    cov: Array

    @property
    def nat_param(self) -> GaussianNatParam:
        chol = jnp.linalg.cholesky(self.cov)
        eye = jnp.eye(self.cov.shape[-1], dtype=self.cov.dtype)
        precision = solve_triangular(chol.T, solve_triangular(chol, eye, lower=True), lower=False)
        return GaussianNatParam(precision, precision @ self.mean)

    def sample(self, key: Array, shape: Sequence[int] = ()) -> Array:
        return jr.multivariate_normal(key, self.mean, self.cov, shape=shape)

    def log_pdf(self, value: Array):
        chol = jnp.linalg.cholesky(self.cov)
        return mvn_logpdf(value, self.mean, chol, constant=True)

    def tree_flatten(self):
        return (self.mean, self.cov), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    