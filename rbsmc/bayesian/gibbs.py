"""
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

from jax import Array
from jax.random import PRNGKey

from rbsmc.bayesian.dists import NatParam


@dataclass(frozen=True)
class GibbsContext:
    trajectory: Array
    dts: Array
    data: Any
    params: dict[str, Any]


class GibbsBlock(ABC):
    names: tuple[str, ...]

    @abstractmethod
    def init(self, key: PRNGKey, hyperparams: dict | None):
        pass

    @abstractmethod
    def sample(self, key: PRNGKey, context: GibbsContext) -> dict[str, Any]:
        pass


#####################################
#       Full-Conditional Gibbs      #
#####################################

@dataclass(frozen=True)
class ConjugateBlock(GibbsBlock):
    name: str
    prior: NatParam | Callable[[dict[str, Any]], NatParam]
    likelihood: Callable[[GibbsContext], NatParam]
    unpack: Callable[[Array], dict[str, Any]]
    shape: tuple = ()

    def _get_prior(self, params: dict[str, Any]) -> NatParam:
        return self.prior(params) if callable(self.prior) else self.prior

    def init(self, key: PRNGKey, params: dict[str, Any]):
        prior = self._get_prior(params)
        sample = prior.dist_param.sample(key, self.shape)
        return self.unpack(sample)

    def sample(self, key: PRNGKey, context: GibbsContext) -> dict[str, Any]:
        prior = self._get_prior(context.params)
        posterior = prior + self.likelihood(context)
        sample = posterior.dist_param.sample(key, self.shape)
        return self.unpack(sample)


@dataclass(frozen=True)
class ConditionalBlock(GibbsBlock):
    names: tuple[str, ...]
    initialiser: Callable[[PRNGKey], dict[str, Any]]
    kernel: Callable[[PRNGKey, GibbsContext], dict[str, Any]]

    def init(self, key, *args, **kwargs):
        return self.initialiser(key)

    def sample(self, key: PRNGKey, context: GibbsContext):
        return self.kernel(key, context)


class Gibbs:

    def __init__(self, blocks: list[GibbsBlock]):
        self.blocks = blocks

    def init(self, key: PRNGKey, fixed_params: dict):
        keys = jr.split(key, len(self.blocks))

        params = dict(fixed_params)
        for _key, block in zip(keys, self.blocks):
            params = {**params, **block.init(_key, params)}
        return params

    def update(self, key: PRNGKey, params: dict, trajectory: Array, dts: Array, data):
        """ Run a set of sequential gibbs samples """
        keys = jr.split(key, len(self.blocks))

        new_params = params
        for _key, _block in zip(keys, self.blocks):
            context = GibbsContext(trajectory=trajectory, dts=dts, data=data, params=new_params)
            new_params = {**new_params, **_block.sample(_key, context)}
        return new_params
    