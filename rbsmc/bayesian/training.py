"""

Write trainer like in RPM-SLAM, to retrieve the loss from a jitted free_energy function. 
Then run optax updates
"""
from dataclasses import dataclass, field
from typing import Callable, Union, Optional, Tuple
from tqdm import tqdm
from copy import deepcopy

import numpy as np

import optax
import jax
from jax import Array
from jax import tree_util

import jax.random as jr
import jax.numpy as jnp



@dataclass
class Config:
    """
    Training configuration for recognition and prior optimisation.
    """
    samples: int = 1000
    burnin: int = 1000
    seed: int = 0

    replacement_rate_window: int = 100

    debug: bool = False


class BayesianInference:

    def __init__(
            self,
            posterior_function: Callable,
            initial_posterior_function: Callable,
            loss_function: Callable,
            prior_init: Callable,
            prior,
            config: Config,
            stabilise_function: Callable | None = None,
        ):
        """
        Parameters
        ----------
        posterior_function:     Callable running posterior sampling from the current model parameters.
        loss_function:          Callable returning the stochastic negative free-energy estimate and auxiliary outputs.
        prior_init:             Callable initialising the prior parameter dictionary.
        prior_sample_init:      Callable initialising latent reference paths or cached samples for the SMC kernel.
        config:                 Training configuration.
        stabilise_function:     Optional callable applied to the full prior parameter dictionary after each update.
        """

        self.loss_hist = None

        self.get_posterior = posterior_function
        self.get_initial_posterior = initial_posterior_function
        self.loss = loss_function
        self.prior_init = prior_init
        self.prior = prior
        self.config = config

        self.stabilise_fn = stabilise_function if stabilise_function is not None else lambda _params: _params

        self._train_data_ndim = None

    def train_step(
            self,
            key,
            params,
            samples: Array,
            data: Union[Array, Tuple[Array]],
        ):
        """
        Runs a single stochastic EM/ECM training step.

        Parameters
        ----------
        key:         PRNG key used by the stochastic free-energy estimator.
        params:      Dictionary with fields:
                        - "prior": prior parameters with trainable grad/exact blocks and fixed parameters.
        samples:     Cached reference paths for the current minibatch.
        data:        Mini-batch of observations.

        Returns
        -------
        loss:            Scalar stochastic negative free-energy estimate.
        aux:             Auxiliary dictionary containing cached samples, SMC ancestor indices, and optional sufficient statistics.
        new_params:      Updated parameter dictionary after the Optax update and optional exact update.
        new_opt_states:  Updated optimiser-state dictionary.
        """
        key_e, key_m = jr.split(key)
        loss, aux = self.loss(key_e, params, samples, data)                      # E-step
        new_params = self.prior.update(key_m, params["prior"], aux["samples"])   # M-step
        return loss, aux, {"prior": new_params}


    def fit(self, data):
        """
        Runs Bayesian Inference.

        Parameters
        ----------
        x0:    The starting point for the MCMC chain. Often an unconditional SMC smoothing sample.
        data:  PyTree of observations with leaves of shape (N, T, *_), where 
                - N is the number of independent time-series
                - T is the number of time-steps

        Returns
        -------
        best_params: Parameter dictionary achieving the lowest observed stochastic loss.
        """
       
        data_leaf = tree_util.tree_leaves(data)[0]
        N, T = data_leaf.shape[:2]
        self._train_data_ndim = data_leaf.ndim
        print(f"Training with N={N}, T={T}")

        train_step = jax.jit(self.train_step) if not self.config.debug else self.train_step
        initial_posterior = jax.jit(self.get_initial_posterior) if not self.config.debug else self.get_initial_posterior
        
        # RNG keys
        key, x0_init_key, prior_init_key = jr.split(jr.PRNGKey(self.config.seed), 3)

        # parameter initialisation
        _prior_params = self.prior_init(prior_init_key)
        self.params = {"prior": _prior_params}

        # state initialisation
        dummy_state = self.prior.dummy_init(params=self.params["prior"], data=data)   # (B, T, D)
        xs0, *_ = initial_posterior(x0_init_key, self.params, dummy_state, data)      # (B, S, T, D)
        x0 = tree_util.tree_map(lambda _x: _x[:, -1, ...], xs0)                       # (B, T, D)
        
        # stores
        self.best_params = None
        self.best_loss = float("inf")
        self.loss_hist = []

        self.replaced_hist = jnp.zeros((T, self.config.replacement_rate_window)) * jnp.nan
        self.replacement_rates = []

        # number of MCMC chain steps
        num_iter = self.config.burnin + self.config.samples

        # store history of parameters and samples
        self.param_hist = deepcopy(self.params)
        self.sample_hist = tree_util.tree_map(lambda _s: _s[None, ...], x0)  # prepend itr dimension

        # run
        pbar = tqdm(range(num_iter))
        for self.itr in pbar:
            key, sample_key = jr.split(key)

            loss, aux, self.params = train_step(
                sample_key,
                self.params,
                tree_util.tree_map(lambda _s: _s[-1, ...], self.sample_hist),
                data,
            )

            # track loss
            loss_float = float(loss)
            self.loss_hist.append(loss_float)
            pbar.set_postfix(loss=f"{loss_float:.3f}")

            if loss_float < self.best_loss:
                self.best_loss = loss_float
                self.best_params = self.params

            # update params and append history
            self.params = {**self.params, "prior": self.stabilise_fn(self.params["prior"])}
            self.param_hist = self._append_param_hist(self.itr, self.param_hist, self.params)
            self.sample_hist = self._append_sample_hist(self.sample_hist, aux["samples"])

            # track replacement rate
            replacement_rates = self._calculate_replacement_rate(aux)
            self.replacement_rates.append(replacement_rates)

        return self.param_hist
    
    # def initialise_sample_cache(self, key, data):
    #     """
    #     Produces rough initial latent reference paths for a dataset or minibatch.

    #     Parameters
    #     ----------
    #     key:  RNG key used by the sample initialiser.
    #     data: Observation PyTree with leaves of shape (N, T, *_).
    #     """
    #     return self.prior.sample_init(key=key, params=self.params["prior"], data=data)
    
    def _calculate_replacement_rate(self, aux: dict):
        """
        Calculate the replacement rate of SMC kernel over a window of sample time
        """
        ancestors = aux["As"]             # (B, T)
        next_ancestors = aux["next_As"]   # (B, S, T)

        replaced = ancestors[:, None, :] != next_ancestors    # (B, S, T)
        replaced = jnp.mean(replaced, axis=(0, 1))            # (T,)

        self.replaced_hist = self.replaced_hist.at[:, 1:].set(self.replaced_hist[:, :-1])  # (T, Window)
        self.replaced_hist = self.replaced_hist.at[:, 0].set(replaced)                     # (T, Window)
        
        replacement_rates = jnp.nanmean(self.replaced_hist, 1) # (T,)
        return replacement_rates
    
    def _append_param_hist(self, itr: int, hist: dict, params: dict):
        """
        Appends the current trainable prior parameters along a leading
        iteration axis while leaving fixed parameters unchanged.
        """
        hist_trainable = hist["prior"]["trainable"]
        trainable = params["prior"]["trainable"]

        # append the latest value to the existing history axis
        if itr == 0:
            hist_trainable = tree_util.tree_map(
                lambda old, new: jnp.stack((old, new), axis=0),
                hist_trainable,
                trainable,
            )
        else:
            hist_trainable = tree_util.tree_map(
                lambda old, new: jnp.concatenate((old, new[None, ...]), axis=0),
                hist_trainable,
                trainable,
            )

        return {**hist, "prior": {**hist["prior"], "trainable": hist_trainable,}}
    
    def _append_sample_hist(self, hist, sample):
        """
        
        Parameters
        ----------
        hist:    Pytree (itr, B, T, *D) - current total sample history
        sample:  Pytree (B, T, *D)      - single sample from SMC with current prior params

        Returns
        -------
        new_hist:  Extended sample history
        """
        return tree_util.tree_map(lambda old, new: jnp.concatenate((old, new[None, ...]), axis=0), hist, sample)
    
    # def apply(
    #         self, 
    #         key,
    #         data,
    #         inits: Array | None,
    #         num_iter: int = 1,
    #     ):
    #     """
    #     Runs posterior sampling on new data.

    #     Parameters
    #     ----------
    #     key:       RNG key used for initialisation and posterior sampling.
    #     data:      Observation PyTree with leaves of shape (B, T, *_).
    #     inits:     Optional initial reference paths. If None, these are initialised from sample_init.
    #     num_iter:  Number of posterior sampler calls.

    #     Returns
    #     -------
    #     samples:  Posterior samples from the final posterior call.
    #     """
    #     assert num_iter >= 1, "num_iter must be at least 1 for Trainer.apply"

    #     key, init_key = jr.split(key)
    #     samples = self.initialise_sample_cache(init_key, data) if inits is None else inits

    #     _get_posterior = lambda _key, _smpls, _data: self.get_posterior(_key, self.params, _smpls, _data)
    #     _get_posterior = jax.jit(_get_posterior)

    #     keys = jr.split(key, num_iter)
    #     pbar = tqdm(range(num_iter - 1), desc="Apply")
    #     for itr in pbar:
    #         _key = keys[itr]
    #         samples, *_ = _get_posterior(_key, samples, data)
    #         samples = tree_util.tree_map(lambda z: z[:, -1], samples)

    #     return _get_posterior(keys[-1], samples, data)[0]
