"""

I really want to represent the conjugate updates as the sum over natural parameters
A new version of Bayesian inference calls a single Gibbs update function.

Lets ignore the Horseshoe in the room right now. 
Lets just say I have a Gaussian prior over the transition matrix A.
I have the likelihood which is the product over data distributed N(Ax_{t-1}, C)
This can be represented as a LinearGaussianChain whose NatParam is the sum from t=0
to T of the NatParams of the individual Gaussian factors. Thus I should just be able 
to calculate the posterior for A as something like prior + likelihood (chain). 
So I would need to define a GibbsUpdate that took a prior and likelihood function, 
just like before. This time though, to handle the Horseshoe stuff I can just write
HorseshoeUpdate(GibbsBlock), define a sample method that runs the horseshoe update
I dont necessarily need the natural parameter structure for it used in ConjugateBlock. 

Now heres the 

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

from deprecated.dists import NatParam
from deprecated.smc import SMC, SMCPosterior


@dataclass(frozen=True)
class GibbsContext:
    trajectory: Array
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


@dataclass(frozen=True)
class ConjugateBlock(GibbsBlock):
    name: str
    prior: NatParam | Callable[[dict[str, Any]], NatParam]
    likelihood: Callable[[GibbsContext], NatParam]
    unpack: Callable[[Array], dict[str, Any]]

    def init(self, key: PRNGKey, hyperparams: dict | None):
        prior = self.prior(hyperparams) if callable(self.prior) else self.prior
        return {self.name: prior.dist_param.sample(key)}

    def sample(self, key: PRNGKey, context: GibbsContext) -> dict[str, Any]:
        prior = self.prior(context.params) if callable(self.prior) else self.prior
        posterior = prior + self.likelihood(context)
        return self.unpack(posterior.dist_param.sample(key))


@dataclass(frozen=True)
class ConditionalBlock(GibbsBlock):
    names: tuple[str, ...]
    kernel: Callable[[Array, GibbsContext], dict[str, Any]]

    def sample(self, key: PRNGKey, context: GibbsContext) -> dict[str, Any]:
        return self.kernel(key, context)


class Gibbs:

    def __init__(self, blocks: list[GibbsBlock]):
        self.blocks = blocks

    def init(self, key, hyperparams: dict):
        """
        I want to use Gibbs to initialise parameter values by sampling from the prior
        for each parameter. This involves updates having access to a salient prior? Which
        might mess this whole thing up - because currently we adaptively construct prior
        as a function of other parameters. This is so that gibbs updates can use new parameter
        values sampled during the block updates. Maybe a set of hyperparameters is all thats needed
        and I can use those - since I assume all priors should be available given all hyperparameters.
        """
        keys = jr.split(key, len(self.blocks))

        params = hyperparams
        for _key, _block in zip(keys, self.blocks):
            params = {**params, **_block.init(_key, params)}
        return params

    def update(self, key, params, trajectory, data):
        """ Run a set of sequential gibbs samples """
        keys = jr.split(key, len(self.blocks))

        new_params = params
        for _key, _block in zip(keys, self.blocks):
            context = GibbsContext(trajectory=trajectory, data=data, params=new_params)
            new_params = {**new_params, **_block.sample(_key, context)}
        return new_params
    