"""Particle Gibbs training loop."""
from dataclasses import dataclass
from typing import Union, Tuple

from tqdm import tqdm

import numpy as np

import jax
from jax import Array, tree_util
import jax.random as jr

from rbsmc.bayesian.smc import Reference, SMC
from rbsmc.bayesian.gibbs import Gibbs


@dataclass
class Config:
    """Particle Gibbs training configuration."""

    samples: int = 1000
    burnin: int = 1000
    seed: int = 0

    thin: int = 1
    saved_paths: int | None = None

    replacement_rate_window: int = 100
    debug: bool = False

    def __post_init__(self):
        if isinstance(self.thin, bool) or self.thin < 1:
            raise ValueError("thin must be a positive integer.")

        if self.saved_paths is not None and (isinstance(self.saved_paths, bool) or self.saved_paths < 1):
            raise ValueError("saved_paths must be None or a positive integer.")

        if self.samples < 0:
            raise ValueError("samples must be non-negative.")

        if self.burnin < 0:
            raise ValueError("burnin must be non-negative.")

        if self.replacement_rate_window < 1:
            raise ValueError("replacement_rate_window must be a positive integer.")


class ParticleGibbs:

    def __init__(
            self,
            smc: SMC,
            gibbs: Gibbs,
            config: Config,
        ):
        self.smc = smc
        self.gibbs = gibbs
        self.config = config

        self.thin = config.thin
        self.saved_paths = config.saved_paths

    def train_step(
            self,
            key,
            params,
            state: Reference,
            dts: Array,
            data: Union[Array, Tuple[Array]],
        ):
        """
        Runs a single stochastic EM/ECM training step.

        Parameters
        ----------
        key:      RNG.
        params:   Dictionary containing parameters for prior model.
        samples:  Cached reference path.
        data:     Observations.

        Returns
        -------
        """
        key_e, key_m = jr.split(key)

        # E step
        state, aux = self.smc.sample(key_e, params, state, data)
        energy = 0

        # M step
        new_params = self.gibbs.update(key_m, params, state.trajectory, dts, data)
        return energy, new_params, state, aux

    def run(self, data, dts: Array, hyperparams: dict):
        """
        Runs Bayesian Inference.

        Parameters
        ----------
        data:         PyTree of observations with leaves of shape (T, *D), where 
                          - T is the number of time-steps
                          - *D is the arbitrary observation dimension
        dts:          The set of time-differentials between each observation time
        hyperparams:  Fixed / initial parameters required for prior initialisation
        """
        data_leaf = tree_util.tree_leaves(data)[0]
        T = data_leaf.shape[0]
        total = self.config.burnin + self.config.samples

        train_step = self.train_step if self.config.debug else jax.jit(self.train_step)

        # initialisation
        key, sample_key, param_key = jr.split(jr.PRNGKey(self.config.seed), 3)
        self.params = self.gibbs.init(param_key, hyperparams)
        state = self.smc.init(sample_key, self.params, data)

        # initialise stores
        self.energies = np.empty(total, dtype=np.float32)
        self._allocate_hist(state, self.params, T)

        # run
        pbar = tqdm(range(total))
        for itr in pbar:
            key, subkey = jr.split(key)

            energy, self.params, state, aux = train_step(subkey, self.params, state, dts, data)

            # track energy
            energy_float = float(energy)
            self.energies[itr] = energy_float
            pbar.set_postfix(loss=f"{energy_float:.3f}")

            replacement_rates = self._calculate_replacement_rate(aux["replaced"])

            # store a thinned array for memory constraints
            if itr % self.thin == 0:
                store_idx = itr // self.thin
                self._store_iteration(store_idx, state, self.params, replacement_rates)

        self._finalise_reference_hist()

        return self.reference_hist, self.param_hist, self.replacement_rates

    def _allocate_hist(self, state, params, T):
        total = self.config.burnin + self.config.samples
        num_stored = 0 if total == 0 else (total - 1) // self.thin + 1

        self.param_hist = tree_util.tree_map(
            lambda x: np.empty((num_stored + 1,) + x.shape, dtype=np.asarray(x).dtype),
            params,
        )
        self.param_hist = tree_util.tree_map(
            lambda hist, x: self._set_hist_value(hist, 0, x),
            self.param_hist,
            params,
        )
        self.replacement_rates = np.empty((num_stored, T), dtype=np.float32)

        if self.saved_paths is None:
            self.reference_capacity = num_stored + 1
        else:
            self.reference_capacity = min(self.saved_paths, num_stored + 1)

        self.reference_hist = tree_util.tree_map(
            lambda x: np.empty((self.reference_capacity,) + x.shape, dtype=np.asarray(x).dtype),
            state,
        )
        self.reference_hist = tree_util.tree_map(
            lambda hist, x: self._set_hist_value(hist, 0, x),
            self.reference_hist,
            state,
        )
        self.reference_count = 1
        self.reference_position = 1 % self.reference_capacity

        # Sliding state used to calculate replacement-rate diagnostics.
        window = min(self.config.replacement_rate_window, max(total, 1))
        self.replaced_hist = np.zeros((window, T), dtype=bool)
        self.replaced_count = 0
        self.replaced_position = 0

    def _store_iteration(
            self,
            store_idx,
            state,
            params,
            replacement_rates,
        ):
        """
        Store one retained Gibbs iteration.

        Parameter-history index zero contains initialization, so retained
        Gibbs iteration `store_idx` is written to `store_idx + 1`.
        """
        param_hist_idx = store_idx + 1
        reference_idx = self.reference_position

        self.param_hist = tree_util.tree_map(
            lambda hist, x: self._set_hist_value(hist, param_hist_idx, x),
            self.param_hist,
            params,
        )

        self.replacement_rates[store_idx] = replacement_rates

        # references use a rolling circular buffer
        self.reference_hist = tree_util.tree_map(
            lambda hist, x: self._set_hist_value(hist, reference_idx, x),
            self.reference_hist,
            state,
        )
        self.reference_position = (self.reference_position + 1) % self.reference_capacity
        self.reference_count = min(self.reference_count + 1, self.reference_capacity)

    def _finalise_reference_hist(self):
        """
        Convert the circular trajectory buffer to chronological order.

        After wrapping, reference_position identifies the oldest retained
        trajectory. Reordering is performed only once, after sampling.
        """
        if self.reference_count < self.reference_capacity:
            order = np.arange(self.reference_count)
        else:
            order = (np.arange(self.reference_count) + self.reference_position) % self.reference_capacity

        self.reference_hist = tree_util.tree_map(lambda hist: hist[order], self.reference_hist)
        self.ancestor_hist = self.reference_hist.ancestors

    @staticmethod
    def _set_hist_value(hist, idx, value):
        hist[idx] = np.asarray(value)
        return hist

    def _calculate_replacement_rate(self, replaced):
        """Calculate replacement rates over a moving iteration window."""
        replaced = np.asarray(replaced, dtype=bool)

        self.replaced_hist[self.replaced_position] = replaced

        self.replaced_position = (self.replaced_position + 1) % self.replaced_hist.shape[0]
        self.replaced_count = min(self.replaced_count + 1, self.replaced_hist.shape[0])

        return self.replaced_hist[:self.replaced_count].mean(axis=0)
