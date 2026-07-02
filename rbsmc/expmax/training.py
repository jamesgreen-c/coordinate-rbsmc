"""

Write trainer like in RPM-SLAM, to retrieve the loss from a jitted free_energy function. 
Then run optax updates
"""
from dataclasses import dataclass, field
from typing import Callable, Union, Optional, Tuple
from tqdm import tqdm

import optax
import jax
from jax import Array
from jax import tree_util

import jax.random as jr
import jax.numpy as jnp


LearningRate = Union[float, Callable[[int], float]]


@dataclass
class OptimConfig:
    """ Configuration for a single parameter block optimiser. """
    optimizer: Callable[[float], optax.GradientTransformation] = optax.adam
    lr: LearningRate = 1e-3

    decay_steps: Optional[int] = None
    decay_rate: Optional[float] = None
    staircase: bool = False

    def schedule(self) -> LearningRate:
        """
        Returns either a constant learning rate, a user-provided schedule, or an exponential-decay schedule.
        """
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
        """ Builds the Optax optimiser. """
        return self.optimizer(self.schedule())


@dataclass
class Config:
    """
    Training configuration for recognition and prior optimisation.
    """
    batch_size: int = 32
    num_iter: int = 1000
    seed: int = 0

    replacement_rate_window: int = 100

    debug: bool = False
    sample_with_replacement: bool = False

    prior: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            optimizer=optax.adam,
            lr=1e-3,
        )
    )


class Trainer:

    # TODO remove sample cache - there is no need for it here with coordinate rbsmc
    def __init__(
            self,
            posterior_function: Callable,
            loss_function: Callable,
            prior_init: Callable,
            prior_sample_init: Callable,
            config: Config,
            exact_update_function: Callable | None = None,
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
        exact_update_function:  Optional callable implementing model-specific exact/closed-form parameter updates. 
        stabilise_function:     Optional callable applied to the full prior parameter dictionary after each update.
        """


        self.loss_hist = None

        self.get_posterior = posterior_function
        self.loss = loss_function
        self.prior_init = prior_init
        self.sample_init = prior_sample_init
        self.config = config

        self.exact_update_fn = exact_update_function
        self.stabilise_fn = stabilise_function if stabilise_function is not None else lambda _params: _params

        self.prior_opt = config.prior.build()

        self._train_data_ndim = None

    def train_step(
            self,
            key,
            params,
            opt_states,
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
        opt_states:  Dictionary with fields:
                        - "prior": optimiser state for params["prior"]["trainable"]["grad"].
        samples:     Cached reference paths for the current minibatch.
        data:        Mini-batch of observations.

        Returns
        -------
        loss:            Scalar stochastic negative free-energy estimate.
        aux:             Auxiliary dictionary containing cached samples, SMC ancestor indices, and optional sufficient statistics.
        new_params:      Updated parameter dictionary after the Optax update and optional exact update.
        new_opt_states:  Updated optimiser-state dictionary.
        """

        # get loss and gradients
        (loss, aux), grads = jax.value_and_grad(self.loss, argnums=1, has_aux=True)(key, params, samples, data)

        # collect stochastic updates
        prior_updates, prior_opt_state = self.prior_opt.update(grads["prior"]["trainable"]["grad"], opt_states["prior"], params["prior"]["trainable"]["grad"],)

        # apply stochastic parameter updates
        new_opt_states = {"prior": prior_opt_state}
        new_params = {
            "prior": {
                "trainable": {
                    "grad": optax.apply_updates(params["prior"]["trainable"]["grad"], prior_updates),
                    "exact": params["prior"]["trainable"]["exact"] 
                },
                "fixed": params["prior"]["fixed"],
            },
        }

        # apply any exact parameter updates
        if self.exact_update_fn is not None:
            new_params = {
                "prior": {
                    "trainable": {
                        "grad": new_params["prior"]["trainable"]["grad"],
                        "exact": self.exact_update_fn(params["prior"], aux, data) 
                    },
                    "fixed": new_params["prior"]["fixed"],
                },
            }

        return loss, aux, new_params, new_opt_states


    def fit(self, data):
        """
        Runs stochastic EM/ECM training.

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
        self._train_data_ndim = data_leaf.ndim
        print(f"Training with N={N}, T={T}")

        train_step = jax.jit(self.train_step) if not self.config.debug else self.train_step

        # RNG keys
        key, prior_init_key = jr.split(jr.PRNGKey(self.config.seed), 2)

        # initialisation
        _prior_params = self.prior_init(prior_init_key)

        self.params = {"prior": _prior_params}
        self.opt_states = {"prior": self.prior_opt.init(self.params["prior"]["trainable"]["grad"])}

        # stores
        self.best_params = None
        self.best_loss = float("inf")
        self.loss_hist = []

        self.replaced_hist = jnp.zeros((T, self.config.replacement_rate_window)) * jnp.nan
        self.replacement_rates = []

        # batching
        batch_size = min(N, self.config.batch_size)
        batch_idx = self.get_batching_function(batch_size, N)

        # sample cache
        key, cache_key = jr.split(key)
        sample_cache = self.initialise_sample_cache(cache_key, data)

        # run
        pbar = tqdm(range(self.config.num_iter))
        for self.itr in pbar:
            key, batch_key, sample_key = jr.split(key, 3)

            idx = batch_idx(batch_key)
            batch = tree_util.tree_map(lambda z: z[idx], data)
            batch_sample_cache = tree_util.tree_map(lambda z: z[idx], sample_cache)

            loss, aux, self.params, self.opt_states = train_step(
                sample_key,
                self.params,
                self.opt_states,
                batch_sample_cache,
                batch,
            )

            # track loss
            loss_float = float(loss) / batch_size
            self.loss_hist.append(loss_float)
            pbar.set_postfix(loss=f"{loss_float:.3f}")

            if loss_float < self.best_loss:
                self.best_loss = loss_float
                self.best_params = self.params

            # update params
            self.params = {
                **self.params,
                "prior": self.stabilise_fn(self.params["prior"])
            }

            # update sample cache
            samples = aux["samples"]
            sample_cache = tree_util.tree_map(lambda cache, new: cache.at[idx].set(new), sample_cache, samples)

            # track replacement rate
            replacement_rates = self._calculate_replacement_rate(aux)
            self.replacement_rates.append(replacement_rates)

        return self.best_params
    
    def get_batching_function(self, batch_size: int, N: int):
        """
        Constructs a minibatch index sampler.

        Parameters
        ----------
        batch_size: Number of time series per minibatch.
        N:          Total number of time series.

        Returns
        -------
        batch_idx:  Callable mapping an RNG key to minibatch indices.
        """
        if batch_size == N:
            print("Using entire dataset")
            batch_idx = lambda _: jnp.arange(N)
        else:
            if self.config.sample_with_replacement:
                batch_idx = lambda _k: jr.randint(_k, (batch_size,), 0, N)
            else:
                batch_idx = lambda _k: jr.choice(_k, N, shape=(batch_size,), replace=False)
        return batch_idx
    
    def initialise_sample_cache(self, key, data):
        """
        Produces rough initial latent reference paths for a dataset or minibatch.

        Parameters
        ----------
        key:  RNG key used by the sample initialiser.
        data: Observation PyTree with leaves of shape (B, T, *_) or (N, T, *_).
        """
        return self.sample_init(key=key, params=self.params["prior"], data=data)
    
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
    
    def apply(
            self, 
            key,
            data,
            inits: Array | None,
            num_iter: int = 1,
        ):
        """
        Runs posterior sampling on new data.

        Parameters
        ----------
        key:       RNG key used for initialisation and posterior sampling.
        data:      Observation PyTree with leaves of shape (B, T, *_).
        inits:     Optional initial reference paths. If None, these are initialised from sample_init.
        num_iter:  Number of posterior sampler calls.

        Returns
        -------
        samples:  Posterior samples from the final posterior call.
        """
        assert num_iter >= 1, "num_iter must be at least 1 for Trainer.apply"

        key, init_key = jr.split(key)
        samples = self.initialise_sample_cache(init_key, data) if inits is None else inits

        _get_posterior = lambda _key, _smpls, _data: self.get_posterior(_key, self.params, _smpls, _data)
        _get_posterior = jax.jit(_get_posterior)

        keys = jr.split(key, num_iter)
        pbar = tqdm(range(num_iter - 1), desc="Apply")
        for itr in pbar:
            _key = keys[itr]
            _, samples, *_ = _get_posterior(_key, samples, data)
            samples = tree_util.tree_map(lambda z: z[:, -1], samples)

        return _get_posterior(keys[-1], samples, data)[0]
