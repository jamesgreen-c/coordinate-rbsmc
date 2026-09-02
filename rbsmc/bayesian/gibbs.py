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

    def init(self, key: PRNGKey, hyperparams: dict | None):
        prior = self.prior(hyperparams) if callable(self.prior) else self.prior
        return self.unpack(prior.dist_param.sample(key, self.shape))

    def sample(self, key: PRNGKey, context: GibbsContext) -> dict[str, Any]:
        prior = self.prior(context.params) if callable(self.prior) else self.prior
        posterior = prior + self.likelihood(context)
        return self.unpack(posterior.dist_param.sample(key, self.shape))


@dataclass(frozen=True)
class ConditionalBlock(GibbsBlock):
    names: tuple[str, ...]
    prior: Callable[[Array, dict[str | Any]], dict[str, Any]]
    kernel: Callable[[Array, GibbsContext], dict[str, Any]]

    def init(self, key: PRNGKey, hyperparams):
        return self.prior(key, hyperparams)

    def sample(self, key: PRNGKey, context: GibbsContext) -> dict[str, Any]:
        return self.kernel(key, context)


class Gibbs:

    def __init__(self, blocks: list[GibbsBlock]):
        self.blocks = blocks

    def init(self, key, hyperparams: dict):
        """
        """
        keys = jr.split(key, len(self.blocks))

        params = hyperparams
        for _key, _block in zip(keys, self.blocks):
            params = {**params, **_block.init(_key, params)}
        return params

    def update(self, key, params, trajectory, dts, data):
        """ Run a set of sequential gibbs samples """
        keys = jr.split(key, len(self.blocks))

        new_params = params
        for _key, _block in zip(keys, self.blocks):
            context = GibbsContext(trajectory=trajectory, dts=dts, data=data, params=new_params)
            new_params = {**new_params, **_block.sample(_key, context)}
        return new_params
    