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

from rbsmc.bayesian.smc import SMC
from rbsmc.bayesian.gibbs import Gibbs


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
            smc: SMC,
            gibbs: Gibbs,
            config: Config,
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

        self.smc = smc
        self.gibbs = gibbs
        self.config = config

    def train_step(
            self,
            key,
            params,
            state: Array,
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
        """
        key_e, key_m = jr.split(key)

        # E step
        state, aux = self.smc.sample(key_e, params, state, data)
        energy = 0 # how and where to put energy calculation 

        # M step
        new_params = self.gibbs.update(key_m, params, state[0])
        return energy, new_params, state, aux


    def run(self, data):
        """
        Runs Bayesian Inference.

        Parameters
        ----------
        data:  PyTree of observations with leaves of shape (N, T, *_), where 
                - N is the number of independent time-series
                - T is the number of time-steps

        Returns
        -------
        best_params: Parameter dictionary achieving the lowest observed stochastic loss.
        """
       
        data_leaf = tree_util.tree_leaves(data)[0]
        N, T = data_leaf.shape[:2]

        train_step = jax.jit(self.train_step) if not self.config.debug else self.train_step
        
        # initialisation
        key, sample_key, param_key = jr.split(jr.PRNGKey(self.config.seed), 3)
        self.params = self.gibbs.init(param_key)
        state = self.smc.init(sample_key, self.params, data)

        # stores
        self.energies = []
        self.replaced_hist = jnp.zeros((T, self.config.replacement_rate_window)) * jnp.nan  
        self.replacement_rates = []

        self.param_hist = deepcopy(self.params)
        self.sample_hist = tree_util.tree_map(lambda _s: _s[None, ...], state[0])  # prepend itr dimension
        self.ancestor_hist = state[1][None, ...]                                   # prepend itr dimension

        # run
        pbar = tqdm(range(self.config.burnin + self.config.samples))
        for self.itr in pbar:
            key, subkey = jr.split(key)

            energy, self.params, state, aux = train_step(subkey, self.params, state, data)

            # track energy
            energy_float = float(energy)
            self.energies.append(energy)
            pbar.set_postfix(loss=f"{energy_float:.3f}")

            # store parameter history for traces
            self.param_hist = self._append_param_hist(self.itr, self.param_hist, self.params)
            
            # store trajectory and ancestory history
            self.sample_hist = self._append_state_hist(self.sample_hist, state[0])
            self.ancestor_hist = self._append_state_hist(self.ancestor_hist, state[1])

            # track replacement rate
            replacement_rates = self._calculate_replacement_rate(aux)
            self.replacement_rates.append(replacement_rates)

        return self.sample_hist, self.ancestor_hist, self.param_hist, self.replacement_hist

    def _calculate_replacement_rate(self, aux: dict):
        """
        Calculate the replacement rate of SMC kernel over a window of sample time
        """
        replaced = aux["replaced"]             # (T,)

        # maintain a replacement rate window
        self.replaced_hist = self.replaced_hist.at[:, 1:].set(self.replaced_hist[:, :-1])  # (T, Window)
        self.replaced_hist = self.replaced_hist.at[:, 0].set(replaced)                     # (T, Window)

        # replacement rate calculated over said window
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
    
    def _append_state_hist(self, hist, sample):
        """
        
        Parameters
        ----------
        hist:    Pytree (itr, T, *D) - current total sample history
        sample:  Pytree (T, *D)      - single sample from SMC with current prior params

        Returns
        -------
        new_hist:  Extended sample history
        """
        return tree_util.tree_map(lambda old, new: jnp.concatenate((old, new[None, ...]), axis=0), hist, sample)
    