import optax

from typing import Any

from jax import vmap, Array
from jax.lax import stop_gradient as stopgrad

import jax.random as jr
from jax.random import PRNGKey
from jax.tree_util import tree_map

from rbsmc.expmax.training import Config
from rbsmc.smc import SMC, Reference



class FreeEnergy:
    def __init__(self, prior, smc: SMC):
        self.prior = prior
        self.smc = smc

    def configure(self, config: Config):
        """ For easier use in train_continue() """
        self.num_samples = config.num_samples

    def init(
            self,
            key: Array,
            data: tuple[Array],
            config: Config
        ) -> tuple[dict, dict[optax.OptState], dict[optax.GradientTransformation]]:
        self.configure(config)
        key_params, key_state = jr.split(key)

        params = self.prior.init(key_params, data, config)
        opts = config.prior.build()
        opt_states = opts.init(params)

        state = self.smc.init(key_state, params, data)
        return params, opt_states, opts, state

    def loss(self, key: PRNGKey, params: dict, state, data: tuple[Array], dts: Array) -> tuple[float, Any]:

        # update optimised parameters with static params for full dict
        params = {**self.prior.params, **params}
        
        ### E-step
        samples = self.get_posterior(key, params, state, data)
        samples = stopgrad(samples)

        ### for M-step
        loss = self.get_loss(params, samples, data, dts)

        # arbitrary state selection for next iteration
        next_state = tree_map(lambda x: x[0], samples)
        replaced = state.ancestors != next_state.ancestors

        aux = {"state": next_state, "replaced": replaced}

        return loss, aux
    
    def get_posterior(self, key: PRNGKey, params: dict, state: Reference, data):
        keys = jr.split(key, self.num_samples)
        samples, _ = vmap(lambda _k: self.smc.sample(_k, params, state, data))(keys)
        return samples.trajectory, samples.ancestors

    def get_loss(self, params, samples, data, dts):
        trajectories = samples.trajectories
        energies = prior_logpdf(self.prior, params, trajectories, data, dts)
        loss = -energies.mean() - self.prior.theta_logpdf(params) # TODO
        return loss


def prior_logpdf(prior, params, samples, data, dts):
    """
    Evaluates the prior log-density over sampled latent trajectories.

    Parameters
    ----------
    samples:       PyTree of posterior samples with leaves of shape (S, T, *_), where
                        - S is the number of posterior samples
                        - T is the number of time-steps
    dts:           Array of time differentials (T-1,)
    prior_params:  Prior model parameters.

    Returns
    -------
    logpdf:  Array of shape (B, S) containing the prior log-density of each sampled path.
    """

    def _one_path_logpdf(path):
        """
        Evaluates the prior log-density for a single sampled latent trajectory.

        Parameters
        ----------
        path:  PyTree with leaves of shape (T, *_), representing one latent trajectory.

        Returns
        -------
        logpdf:  Scalar prior log-density for the sampled trajectory.
        """

        # p0
        x0 = tree_map(lambda z: z[0], path)
        val = prior.log_p0(params, x0)

        # p_t's
        xp = tree_map(lambda z: z[:-1], path)
        x = tree_map(lambda z: z[1:], path)
        trans_vals = vmap(lambda xp_t, x_t, dt: prior.log_pt(params, xp_t, x_t, dt))(xp, x, dts)

        # h_t's
        emission_vals = vmap(lambda x, y: prior.log_ht(params, x, y))(path, data)

        return val + trans_vals.sum() + emission_vals.sum()

    return vmap(_one_path_logpdf)(samples)
