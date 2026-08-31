from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jax.random as jr

from jax import Array
from jax.random import PRNGKey

from rbsmc.bayesian.dists import NatParam, GaussianNatParam
from rbsmc.bayesian.gibbs import GibbsBlock, GibbsContext


class MetropolisWithinGibbs(GibbsBlock, ABC):

    name: str
    prior: NatParam | Callable[[dict[str, Any]], NatParam]
    likelihood: Callable[[GibbsContext], NatParam]

    def init(self, key: PRNGKey, hyperparams: dict | None):
        prior = self.prior(hyperparams) if callable(self.prior) else self.prior
        return {self.name: prior.dist_param.sample(key)}

    @abstractmethod
    def proposal(self, params: dict[str, Any]) -> NatParam:
        """Construct the proposal distribution around the current parameter."""
        pass

    def log_target(self, value: Array, context: GibbsContext) -> Array:

        # set param value to evaluate
        params = {**context.params, self.name: value}

        # reconstruct context
        candidate_context = GibbsContext(
            trajectory=context.trajectory,
            dts=context.dts,
            data=context.data,
            params=params,
        )

        # calculate target logpdf
        prior = self.prior(params) if callable(self.prior) else self.prior
        likelihood = self.likelihood(candidate_context)

        return prior.dist_param.log_pdf(value) + likelihood.dist_param.log_pdf(value)
        
    def accept_reject(
        self,
        key: PRNGKey,
        current: Array,
        proposed: Array,
        forward_proposal: NatParam,
        reverse_proposal: NatParam,
        context: GibbsContext,
    ) -> Array:

        # calculate target logpdfs for current and proposed parameter values
        log_target_current = self.log_target(current, context)
        log_target_proposed = self.log_target(proposed, context)

        # calculate proposal logpdfs for forward and reverse proposal processes
        log_q_forward = forward_proposal.dist_param.log_pdf(proposed)
        log_q_reverse = reverse_proposal.dist_param.log_pdf(current)

        # accept-reject
        log_acceptance_ratio = log_target_proposed - log_target_current + log_q_reverse - log_q_forward
        accept = jnp.log(jr.uniform(key)) < jnp.minimum(log_acceptance_ratio, 0.0)
        return jnp.where(accept, proposed, current)

    def sample(self, key: PRNGKey, context: GibbsContext) -> dict[str, Any]:
        proposal_key, accept_key = jr.split(key)

        # propose new parameter value
        current = context.params[self.name]
        forward_proposal = self.proposal(context.params)
        proposed = forward_proposal.dist_param.sample(proposal_key)

        # construct reversal proposal distribution
        proposed_params = {**context.params, self.name: proposed}
        reverse_proposal = self.proposal(proposed_params)

        # accept or reject new proposed parameter
        value = self.accept_reject(accept_key,
                                   current,
                                   proposed,
                                   forward_proposal,
                                   reverse_proposal,
                                   context)
        return {self.name: value}


# @dataclass(frozen=True)
# class RandomWalkMH(MetropolisWithinGibbs):
#     name: str
#     prior: NatParam | Callable[[dict[str, Any]], NatParam]
#     likelihood: Callable[[GibbsContext], NatParam]
#     scale: Array

#     def proposal(self, params: dict[str, Any]) -> NatParam:
#         return GaussianNatParam.from_mean_cov(
#             params[self.name],
#             self.scale,
#         )