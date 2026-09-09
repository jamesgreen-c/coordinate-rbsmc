from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jax.random as jr

from jax import Array
from jax.random import PRNGKey
from jax.scipy.linalg import solve

from rbsmc.bayesian.dists import NatParam, GaussianNatParam
from rbsmc.bayesian.gibbs import GibbsBlock, GibbsContext


class MetropolisWithinGibbs(GibbsBlock, ABC):

    name: str
    prior: NatParam | Callable[[dict[str, Any]], NatParam]
    likelihood: Callable[[GibbsContext], Array]
    unpack: Callable[[Array], dict[str, Any]]

    def _get_prior(self, params: dict[str, Any]):
        return self.prior(params) if callable(self.prior) else self.prior

    def init(self, key: PRNGKey, params: dict[str, Any]):
        sample = self._get_prior(params).dist_param.sample(key)
        return self.unpack(sample)

    @abstractmethod
    def proposal(self, params: dict[str, Any]) -> NatParam:
        pass

    def log_target(self, value: Array, context: GibbsContext):
        params = {**context.params, **self.unpack(value)}
        candidate_context = GibbsContext(
            trajectory=context.trajectory,
            dts=context.dts,
            data=context.data,
            params=params,
        )

        prior = self._get_prior(params)
        return prior.dist_param.log_pdf(value) + self.likelihood(candidate_context)

    def accept_reject(
            self,
            key: PRNGKey,
            current: Array,
            proposed: Array,
            forward_proposal: NatParam,
            reverse_proposal: NatParam,
            context: GibbsContext,
    ):
        log_acceptance_ratio = (
            self.log_target(proposed, context)
            - self.log_target(current, context)
            + reverse_proposal.dist_param.log_pdf(current)
            - forward_proposal.dist_param.log_pdf(proposed)
        )

        accept = jnp.log(jr.uniform(key)) < jnp.minimum(log_acceptance_ratio, 0.0)
        return jnp.where(accept, proposed, current)

    def sample(self, key: PRNGKey, context: GibbsContext):
        proposal_key, accept_key = jr.split(key)

        current = context.params[self.name]
        forward_proposal = self.proposal(context.params)
        proposed = forward_proposal.dist_param.sample(proposal_key)

        proposed_params = {**context.params, **self.unpack(proposed)}
        reverse_proposal = self.proposal(proposed_params)

        value = self.accept_reject(
            key=accept_key,
            current=current,
            proposed=proposed,
            forward_proposal=forward_proposal,
            reverse_proposal=reverse_proposal,
            context=context,
        )

        return self.unpack(value)


@dataclass(frozen=True)
class RandomWalkMetropolis(MetropolisWithinGibbs):
    name: str
    prior: NatParam | Callable[[dict[str, Any]], NatParam]
    likelihood: Callable[[GibbsContext], Array]
    unpack: Callable[[Array], dict[str, Any]]
    covariance: Array

    def proposal(self, params: dict[str, Any]):
        precision = solve(self.covariance, jnp.eye(self.covariance.shape[0]))
        current = params[self.name]
        return GaussianNatParam(precision=precision, precision_mean=precision @ current)


    
# from __future__ import annotations

# from abc import ABC, abstractmethod
# from collections.abc import Callable
# from dataclasses import dataclass
# from typing import Any

# import jax.numpy as jnp
# import jax.random as jr

# from jax import Array
# from jax.random import PRNGKey

# from rbsmc.bayesian.dists import NatParam, GaussianNatParam
# from rbsmc.bayesian.gibbs import GibbsBlock, GibbsContext


# class MetropolisWithinGibbs(GibbsBlock, ABC):

#     name: str
#     prior: NatParam | Callable[[dict[str, Any]], NatParam]
#     likelihood: Callable[[GibbsContext], NatParam]

#     def init(self, key: PRNGKey, hyperparams: dict | None):
#         prior = self.prior(hyperparams) if callable(self.prior) else self.prior
#         return {self.name: prior.dist_param.sample(key)}

#     @abstractmethod
#     def proposal(self, params: dict[str, Any]) -> NatParam:
#         """Construct the proposal distribution around the current parameter."""
#         pass

#     def log_target(self, value: Array, context: GibbsContext) -> Array:

#         # set param value to evaluate
#         params = {**context.params, self.name: value}

#         # reconstruct context
#         candidate_context = GibbsContext(
#             trajectory=context.trajectory,
#             dts=context.dts,
#             data=context.data,
#             params=params,
#         )

#         # calculate target logpdf
#         prior = self.prior(params) if callable(self.prior) else self.prior
#         likelihood = self.likelihood(candidate_context)

#         return prior.dist_param.log_pdf(value) + likelihood.dist_param.log_pdf(value)
        
#     def accept_reject(
#         self,
#         key: PRNGKey,
#         current: Array,
#         proposed: Array,
#         forward_proposal: NatParam,
#         reverse_proposal: NatParam,
#         context: GibbsContext,
#     ) -> Array:

#         # calculate target logpdfs for current and proposed parameter values
#         log_target_current = self.log_target(current, context)
#         log_target_proposed = self.log_target(proposed, context)

#         # calculate proposal logpdfs for forward and reverse proposal processes
#         log_q_forward = forward_proposal.dist_param.log_pdf(proposed)
#         log_q_reverse = reverse_proposal.dist_param.log_pdf(current)

#         # accept-reject
#         log_acceptance_ratio = log_target_proposed - log_target_current + log_q_reverse - log_q_forward
#         accept = jnp.log(jr.uniform(key)) < jnp.minimum(log_acceptance_ratio, 0.0)
#         return jnp.where(accept, proposed, current)

#     def sample(self, key: PRNGKey, context: GibbsContext) -> dict[str, Any]:
#         proposal_key, accept_key = jr.split(key)

#         # propose new parameter value
#         current = context.params[self.name]
#         forward_proposal = self.proposal(context.params)
#         proposed = forward_proposal.dist_param.sample(proposal_key)

#         # construct reversal proposal distribution
#         proposed_params = {**context.params, self.name: proposed}
#         reverse_proposal = self.proposal(proposed_params)

#         # accept or reject new proposed parameter
#         value = self.accept_reject(accept_key,
#                                    current,
#                                    proposed,
#                                    forward_proposal,
#                                    reverse_proposal,
#                                    context)
#         return {self.name: value}


# # @dataclass(frozen=True)
# # class RandomWalkMH(MetropolisWithinGibbs):
# #     name: str
# #     prior: NatParam | Callable[[dict[str, Any]], NatParam]
# #     likelihood: Callable[[GibbsContext], NatParam]
# #     scale: Array

# #     def proposal(self, params: dict[str, Any]) -> NatParam:
# #         return GaussianNatParam.from_mean_cov(
# #             params[self.name],
# #             self.scale,
# #         )