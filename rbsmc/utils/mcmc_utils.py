import jax
import jax.numpy as jnp

from rbsmc.utils.pbar import progress_bar_scan
from functools import partial


def delta_adaptation_routine(
        key,
        init_xs, init_bs,
        kernel,
        target_acceptance,
        initial_deltas,
        n_steps,
        verbose=False,
        min_delta=1e-12,
        max_delta=1e2,
        min_rate=1e-2,
        window_size=100,
        rate=0.1,
        **_kwargs
):
    """
    Finds the deltas that roughly achieve the target acceptance rate.
    
    It does this by successively applying the MCMC kernel (eg the Particle-RWM kernel) to produce
    a new reference state (x_ref, b_ref). After each application of the kernel, it computes the
    acceptance based on if the reference ancestor indices have changed.

    Acceptance rates are averaged over a sliding window of size `window_size`, and the deltas
    are updated according to the difference between the observed acceptance rate and the target, 
    but only if the acceptance rate is outside a tolerance of 0.05 from the target.

    This is repeated using the new reference path. This can therefore be seen as a form of sampling
    for a set of deltas that achieve the target acceptance rate. These deltas can then be fixed and 
    used in a subsequent MCMC run.
        
    :param key: rng
    :param init_xs: Reference path
    :param init_bs: Reference ancestor indices
    :param kernel: MCMC kernel function
    :param target_acceptance: Target acceptance rate
    :param initial_deltas: Initial deltas, can be scalar or array of shape (T,)
    :param n_steps: Number of iterations delta finding
    :param verbose: Use a progress bar?
    :param min_delta: Minimum delta
    :param max_delta: Maximum delta
    :param min_rate: The minimum rate of adaptation for deltas
    :param window_size: Window Size
    :param rate: Adaptation rate
    :param _kwargs: Additional arguments - unused 
    """

    T = init_xs.shape[0]

    if verbose:
        decorator = progress_bar_scan(n_steps, show=-1)
    else:
        decorator = lambda x: x

    @decorator
    def body(carry, inp):
        state, deltas, accepted_history, *_ = carry
        xs, bs = state
        i, key_i = inp

        # Run kernel
        next_xs, next_bs, *_ = kernel(key_i, state, deltas)

        accepted = next_bs != bs
        accepted_history = accepted_history.at[:, 1:].set(accepted_history[:, :-1])
        accepted_history = accepted_history.at[:, 0].set(accepted)
        acceptance_rates = jnp.nanmean(accepted_history, 1)

        flag = jnp.logical_or(acceptance_rates < target_acceptance - 0.05,
                              acceptance_rates > target_acceptance + 0.05)
        flag &= i > window_size
        rate_i = jnp.maximum(min_rate, rate / (i + 1) ** 0.5)

        deltas_otherwise = deltas + rate_i * deltas * (
                acceptance_rates - target_acceptance) / target_acceptance

        deltas = jnp.where(flag, deltas_otherwise, deltas)

        deltas = jnp.clip(deltas, min_delta, max_delta)
        carry_out = (next_xs, next_bs), deltas, accepted_history, jnp.mean(acceptance_rates)
        return carry_out, None

    initial_deltas = initial_deltas * jnp.ones(init_xs.shape[0])
    initial_accepted_history = jnp.zeros((T, window_size)) * jnp.nan
    init = (init_xs, init_bs), initial_deltas, initial_accepted_history, jnp.mean(initial_accepted_history)
    inps = jnp.arange(n_steps), jax.random.split(key, n_steps)
    (fin_state, fin_deltas, *_), _ = jax.lax.scan(body, init, inps)
    return fin_state, fin_deltas


def sampling_routine(key,
                     init_xs, init_bs,
                     kernel,
                     n_steps,
                     verbose=False,
                     get_samples=True):
    """
    Runs a generic MCMC sampling routine using the provided MCMC kernel.

    At each step, the MCMC kernel (eg Particle-RWM kernel) is applied to the current state
    which produces a new reference state (x_ref, b_ref). If get_samples is set, them we store
    all the samples and return them at the end, otherwise we just return the final state.
        
    :param key: rng
    :param init_xs: Initial reference path
    :param init_bs: Initial reference ancestor indices
    :param kernel: MCMC kernel function (eg Particle-RWM kernel)
    :param n_steps: Number of iterations
    :param verbose: Use a progress bar?
    :param get_samples: Whether to return all samples or just the final state
    """

    if verbose:
        decorator = progress_bar_scan(n_steps, show=-1)
    else:
        decorator = lambda x: x

    @decorator
    def body(carry, inp):
        i, key_op = inp
        xs, bs, show = carry

        # Run kernel
        next_xs, next_bs, next_log_ws, *_ = kernel(key_op, (xs, bs))
        accepted = next_bs != bs
        carry_out = next_xs, next_bs, jnp.mean(accepted)

        return carry_out, (next_xs, accepted) if get_samples else None

    init = init_xs, init_bs, 0.
    inps = jnp.arange(n_steps), jax.random.split(key, n_steps)
    final_xs, out = jax.lax.scan(body, init, inps)
    if get_samples:
        samples, flags = out
        return samples[::int(get_samples)], flags[::int(get_samples)]
    else:
        return final_xs[:2]


def aux_sampling_routine(key,
                     init_xs, init_bs,
                     kernel,
                     n_steps,
                     verbose=False,
                     get_samples=True):
    """
    Runs a generic MCMC sampling routine using the provided MCMC kernel.

    At each step, the MCMC kernel (eg Particle-RWM kernel) is applied to the current state
    which produces a new reference state (x_ref, b_ref). If get_samples is set, them we store
    all the samples and return them at the end, otherwise we just return the final state.
        
    :param key: rng
    :param init_xs: Initial reference path
    :param init_bs: Initial reference ancestor indices
    :param kernel: MCMC kernel function (eg Particle-RWM kernel)
    :param n_steps: Number of iterations
    :param verbose: Use a progress bar?
    :param get_samples: Whether to return all samples or just the final state
    """

    if verbose:
        decorator = progress_bar_scan(n_steps, show=-1)
    else:
        decorator = lambda x: x

    @decorator
    def body(carry, inp):
        i, key_op = inp
        xs, bs, show = carry

        # Run kernel
        next_xs, next_bs, next_log_ws, *_ = kernel(key_op, (xs, bs))
        accepted = next_bs != bs
        carry_out = next_xs, next_bs, jnp.mean(accepted)

        return carry_out, (next_xs, next_bs, next_log_ws, accepted) if get_samples else None

    init = init_xs, init_bs, 0.
    inps = jnp.arange(n_steps), jax.random.split(key, n_steps)
    final_xs, out = jax.lax.scan(body, init, inps)
    if get_samples:
        return out
    else:
        return final_xs[:2]