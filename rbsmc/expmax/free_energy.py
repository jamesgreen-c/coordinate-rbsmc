from typing import Callable
import jax
import jax.random as jr
import jax.numpy as jnp
from jax import tree_util

from jax import vmap, Array
from jax.random import PRNGKey
from jax.lax import stop_gradient


def constructor(
        prior,
        smc_init: Callable,
        smc: Callable,
        dts: Array,
        n_samples: int = 1,
    ):
    """
    Constructs the prior-only Monte Carlo free-energy loss and posterior sampler.

    Parameters
    ----------
    prior:      Prior/generative model module with:
                    - log_p0:  Evaluate the initial latent density.
                    - log_pt:  Evaluate the transition density.
                    - log_gt:  Evaluate the observation/emission density.
    smc_init:   Initialisation function for the SMC kernel.
    smc:        SMC kernel used to produce posterior samples.
    n_samples:  Number of posterior samples S to draw for each independent time-series.

    Returns
    -------
    posterior: Callable for sampling posterior trajectories using the current prior parameters.
    loss:      Callable returning the negative Monte Carlo free energy and auxiliary SMC outputs.
    """

    # vmapped over batch dimension
    smc = vmap(smc, in_axes=(0, 0, None, 0))
    smc_init = vmap(smc_init, in_axes=(0))

    def posterior(
            key,
            params,
            samples: Array,
            data: tuple,
        ):
        """
        Runs the SMC posterior sampler from cached reference paths.

        Parameters
        ----------
        key:      PRNG key used to run the SMC kernels.
        params:   Parameter dictionary with fields:
                    - "prior": prior/generative model parameters.
        samples:  Cached reference paths or latent states used to initialise the SMC kernels.
        data:     PyTree of observations with leaves of shape (B, T, *_), where B is the number of independent time-series and T is the number of time-steps.

        Returns
        -------
        samples:  PyTree of posterior samples with leaves of shape (B, S, T, *_), where S is the number of posterior samples per time-series.
        As:       Ancestor indices for the cached reference paths before the SMC update.
        next_As:  Ancestor indices for the posterior samples produced by the SMC update.
        """
        # housekeeping
        prior_params = params["prior"]

        data_leaf = jax.tree_util.tree_leaves(data)[0]
        B, T = data_leaf.shape[:2]
        keys = jr.split(key, (B, n_samples))

        # initialise reference path/state for cSMC
        inits = smc_init(samples)
        As = inits[1]

        # get samples from cSMC
        next_samples, next_As, *_= smc(
            keys,
            inits,
            prior_params,
            data,         # mapped over batch
        )

        return next_samples, As, next_As

    def prior_logpdf(samples, data, prior_params):
        """
        Evaluates the complete-data log-density over sampled latent trajectories.

        Parameters
        ----------
        samples:       PyTree of posterior samples with leaves of shape (B, S, T, *_), where
                         - B is the number of independent time-series
                         - S is the number of posterior samples
                         - T is the number of time-steps
        data:          PyTree of observations with leaves of shape (B, T, *_), where 
                         - B is the number of independent time-series
                         - T is the number of time-steps
        prior_params:  Prior/generative model parameters.

        Returns
        -------
        logpdf: Array of shape (B, S) containing the complete-data log-density of each sampled path and observation sequence.
        """

        def _one_path(x_path, ys):
            """
            Evaluates the complete-data log-density for one sampled latent trajectory and one observation sequence.

            Parameters
            ----------
            x_path: PyTree with leaves of shape (T, *_), representing one latent trajectory.
            ys:     PyTree with leaves of shape (T, *_), representing one observation sequence.

            Returns
            -------
            logpdf: Scalar complete-data log-density for the sampled trajectory and observations.
            """
            # p0
            x0 = tree_util.tree_map(lambda z: z[0], x_path)
            val = prior.log_p0(prior_params, x0)

            # p_t's
            xp = tree_util.tree_map(lambda z: z[:-1], x_path)
            x = tree_util.tree_map(lambda z: z[1:], x_path)
            trans_vals = jax.vmap(lambda xp_t, x_t, dt_t: prior.log_pt(prior_params, xp_t, x_t, dt_t))(xp, x, dts[1:])

            # g_t's
            obs_vals = jax.vmap(lambda x_t, y_t: prior.log_potential(prior_params, x_t, y_t))(x_path, ys)

            return val + trans_vals.sum() + obs_vals.sum()

        _one_sample = vmap(_one_path, in_axes=(0, None))   # vmapped over S
        _one_batch = vmap(_one_sample)                     # vmapped over B
        return _one_batch(samples, data)

    def loss(key, params, samples_prev, data):
        """
        Calculates the negative Monte Carlo free-energy estimate.

        Parameters
        ----------
        key:          PRNG key used by the SMC posterior sampler.
        params:       Parameter dictionary with fields:
                        - "prior": prior/generative model parameters
        samples_prev: Cached reference paths or latent states for the current minibatch.
        data:         PyTree of observations with leaves of shape (B, T, *_), where 
                        - B is the number of independent time-series
                        - T is the number of time-steps

        Returns
        -------
        loss: Scalar negative Monte Carlo free-energy estimate.
        aux:  Dictionary containing cached samples, SMC ancestor indices, and any quantities required by diagnostics or exact update blocks.
        """
        prior_params = params["prior"]

        samples, As_p, As = posterior(
            key=key,
            params=params,
            samples=samples_prev,
            data=data,
        )
        samples = stop_gradient(samples)

        val = prior_logpdf(samples, data, prior_params)
        # print(f"Prior logpdf shape: ", val.shape)
        loss = -val.mean(axis=-1).sum()
        
        samples = tree_util.tree_map(lambda s: s[:, -1], samples)   # only return last sample per batch idx B for iCSMC methods
        aux = {"samples": samples, "As": As_p, "next_As": As}
        return loss, aux 
    
    return posterior, loss
