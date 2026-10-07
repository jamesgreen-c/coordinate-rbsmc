from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jax.random as jr

from jax import Array, vmap
from jax.random import PRNGKey
from jax.scipy.linalg import solve

from rbsmc.dists import NatParam, GaussianNatParam
from rbsmc.bayesian.gibbs import GibbsBlock, GibbsContext


class MetropolisWithinGibbs(GibbsBlock, ABC):

    name: str
    prior: NatParam | Callable[[dict[str, Any]], NatParam]
    likelihood: Callable[[GibbsContext], Array]
    unpack: Callable[[Array], dict[str, Any]]

    coordinatewise: bool = False

    def _get_prior(self, params: dict[str, Any]):
        return self.prior(params) if callable(self.prior) else self.prior

    def init(self, key: PRNGKey, params: dict[str, Any]):
        sample = self._get_prior(params).dist_param.sample(key)
        return self.unpack(sample)

    @abstractmethod
    def proposal(self, params: dict[str, Any]) -> NatParam:
        pass

    # def log_target(self, value: Array, context: GibbsContext):
    #     params = {**context.params, **self.unpack(value)}
    #     candidate_context = GibbsContext(
    #         trajectory=context.trajectory,
    #         dts=context.dts,
    #         data=context.data,
    #         params=params,
    #     )

    #     prior = self._get_prior(params)
    #     return prior.dist_param.log_pdf(value) + self.likelihood(candidate_context)

    def log_target(self, value: Array, context: GibbsContext):
        params = {**context.params, **self.unpack(value)}
        candidate_context = GibbsContext(
            trajectory=context.trajectory,
            dts=context.dts,
            data=context.data,
            params=params,
        )

        prior = self._get_prior(params)
        likelihood = self.likelihood(candidate_context)

        if self.coordinatewise:

            if likelihood.shape != value.shape:
                raise ValueError("Coordinate-wise MH requires one likelihood contribution per coordinate.")
               
            log_prior = vmap(
                lambda i: jnp.sum(prior.dist_param.marginal(i).log_pdf(value[i][None]))
            )(jnp.arange(value.size))

            return log_prior + likelihood
        
        return prior.dist_param.log_pdf(value) + jnp.sum(likelihood)

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

    def accept_reject(
            self,
            key: PRNGKey,
            current: Array,
            proposed: Array,
            forward_proposal: NatParam,
            reverse_proposal: NatParam,
            context: GibbsContext,
    ):
        """
        Accept jointly, or independently for factorised targets and proposals.

        Coordinate-wise mode requires independent prior/proposal coordinates
        and a likelihood returning their separate log-density contributions.
        """
        if self.coordinatewise and (current.ndim != 1 or proposed.shape != current.shape):
            raise ValueError("Coordinate-wise MH requires matching parameter vectors.")

        def _accept(key, current, proposed, log_current, log_proposed, forward, reverse):
            log_acceptance_ratio = (
                log_proposed - log_current
                + jnp.sum(reverse.log_pdf(current))
                - jnp.sum(forward.log_pdf(proposed))
            )
            accept = jnp.log(jr.uniform(key)) < jnp.minimum(log_acceptance_ratio, 0.0)
            return jnp.where(accept, proposed, current)

        log_current = self.log_target(current, context)
        log_proposed = self.log_target(proposed, context)
        forward_proposal = forward_proposal.dist_param
        reverse_proposal = reverse_proposal.dist_param

        if self.coordinatewise:

            keys = jr.split(key, current.size)
            _one = lambda i, key: _accept(
                key, 
                current[i][None], 
                proposed[i][None], 
                log_current[i], 
                log_proposed[i], 
                forward_proposal.marginal(i), 
                reverse_proposal.marginal(i)
            )[0]
            return vmap(_one)(jnp.arange(current.size), keys)

        return _accept(
            key, 
            current, 
            proposed, 
            log_current, 
            log_proposed,
            forward_proposal, 
            reverse_proposal,
        )

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

    coordinatewise: bool = False

    def proposal(self, params: dict[str, Any]):
        precision = solve(self.covariance, jnp.eye(self.covariance.shape[0]))   # TODO only needs to be done once
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