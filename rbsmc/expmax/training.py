"""Particle Gibbs training loop."""
from dataclasses import dataclass, field
from typing import Union, Tuple, Optional, Callable

from tqdm import tqdm

import numpy as np

import jax
import jax.random as jr
import optax

from jax import Array, tree_util

from rbsmc.expmax.free_energy import FreeEnergy
from rbsmc.smc import Reference


LearningRate = Union[float, Callable[[int], float]]

@dataclass
class OptimConfig:
    """Configuration for a single parameter-block optimiser."""

    optimizer: Callable[[LearningRate], optax.GradientTransformation] = optax.adam
    lr: LearningRate = 1e-3
    max_grad_norm: Optional[float] = None

    decay_steps: Optional[int] = None
    decay_rate: Optional[float] = None
    staircase: bool = False

    def schedule(self) -> LearningRate:
        if callable(self.lr):
            return self.lr

        if self.decay_steps is None or self.decay_rate is None:
            return self.lr

        return optax.exponential_decay(
            init_value=self.lr,
            transition_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=self.staircase,
        )

    def build(self) -> optax.GradientTransformation:
        transforms = []

        if self.max_grad_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.max_grad_norm))

        transforms.append(self.optimizer(self.schedule()))
        return optax.chain(*transforms)


@dataclass
class Config:
    """Monte Carlo EM training configuration."""

    num_iter: int = 1000
    num_samples: int = 1
    seed: int = 0

    prior: OptimConfig = field(default_factory=lambda: OptimConfig(
        optimizer=optax.adam,
        lr=1e-3,
        max_grad_norm=10.0,
    ))

    thin: int = 1
    saved_paths: int | None = None

    replacement_rate_window: int = 100
    debug: bool = False

    def __post_init__(self):
        if isinstance(self.thin, bool) or self.thin < 1:
            raise ValueError("thin must be a positive integer.")

        if self.saved_paths is not None and (isinstance(self.saved_paths, bool) or self.saved_paths < 1):
            raise ValueError("saved_paths must be None or a positive integer.")

        if self.num_iter < 0:
            raise ValueError("num_iter must be non-negative.")

        if self.num_samples < 1:
                    raise ValueError("num_iter must be positive.")

        if self.replacement_rate_window < 1:
            raise ValueError("replacement_rate_window must be a positive integer.")


class MonteCarloEM:

    def __init__(
            self,
            model: FreeEnergy,
            config: Config,
        ):
        self.model = model
        self.config = config

        self.thin = config.thin
        self.saved_paths = config.saved_paths

    def train_step(
            self,
            key,
            params,
            opt_states,
            state,
            data: Union[Array, Tuple[Array]],
            dts: Array
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
        # E-step
        (loss, aux), grads = jax.value_and_grad(self.model.loss, has_aux=True, argnums=1)(
            key, params, state, data, dts
        )

        # M-step
        updates, new_opt_states = self.opts.update(grads, opt_states, params)                                  
        new_params = optax.apply_updates(params, updates)
        return -loss, new_params, new_opt_states, aux

    def run(self, data, dts: Array, hyperparams: dict):
        """
        Runs Monte Carlo Expectation Maximisation.

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

        train_step = self.train_step if self.config.debug else jax.jit(self.train_step)

        # initialisation
        key, init_key = jr.split(jr.PRNGKey(self.config.seed))
        self.params, self.opt_states, self.opts, state = self.model.init(
            init_key, 
            data, 
            self.config, 
            hyperparams
        )

        # initialise stores
        self.energies = np.empty(self.config.num_iter, dtype=np.float32)
        self._allocate_hist(state, self.params, T)

        # run
        pbar = tqdm(range(self.config.num_iter))
        for itr in pbar:
            key, subkey = jr.split(key)

            energy, self.params, self.opt_states, aux = train_step(subkey, self.params, self.opt_states, state, data, dts)
            state = aux["state"]

            # track energy
            energy_float = float(energy)
            self.energies[itr] = energy_float
            pbar.set_postfix(energy=f"{energy_float:.3f}")
            
            # calculate replacement rate
            replacement_rates = self._calculate_replacement_rate(aux["replaced"])

            # store a thinned array for memory constraints
            if itr % self.thin == 0:
                store_idx = itr // self.thin
                self._store_iteration(store_idx, state, self.params, replacement_rates)

        self._finalise_reference_hist()

        return self.reference_hist, self.param_hist, self.replacement_rates

    def _allocate_hist(self, state, params, T):
        total = self.config.num_iter
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

        # sliding state used to calculate replacement-rate diagnostics.
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
