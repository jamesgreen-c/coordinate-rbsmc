"""
Implementation of conditional multinomial and killing resampling algorithms.
"""
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from jax.scipy.special import logsumexp


def dynamic(resampling_fn: Callable, threshold: float, key: PRNGKey, weights: Array, i: int, j: int, conditional: bool = True) -> Array:
    """
    Dynamic resampling. The weights are assumed to be normalised already.
    This function should be closed over with a predefined resampling function and threshold: 

        e.g. 
        threshold = 0.5
        resampler = multinomial
        def resampling_fn(key, weights, i, j, conditional):
            return dynamic(resampler, threshold, key, weights, i, j, conditional)
    
    Parameters
    ----------
    key:
        Random number generator key.
    weights:
        Weights of the particles.
    i, j:
        Conditional indices: the resampling is conditioned on the fact that the ancestor at index j is equal to i.
    conditional:
        If True, the resampling is conditional on the fact that the ancestor at index j is equal to i.
        Otherwise, it's the standard resampling
    threshold:
        Is a fraction defining the threshold ESS (against the number of particles) that must be met before resampling.
        
    Returns
    -------
    indices:
        Indices of the resampled particles.
    """
    N = weights.shape[0]
    # jax.debug.print("resampling weights shape: {}", weights.shape)    # output: (N,)
    _ess = 1.0 / jnp.sum(weights**2)

    def resample(_):
        return resampling_fn(key, weights, i, j, conditional)

    def skip(_):
        return jnp.arange(N)

    return jax.lax.cond(
        _ess < N * threshold,
        resample,
        skip,
        operand=None,
    )


def multinomial(key: PRNGKey, weights: Array, i: int, j: int, conditional: bool = True) -> Array:
    """
    Conditional multinomial resampling. The weights are assumed to be normalised already.

    Parameters
    ----------
    key:
        Random number generator key.
    weights:
        Weights of the particles.
    i, j:
        Conditional indices: the resampling is conditioned on the fact that the ancestor at index j is equal to i.
    conditional:
        If True, the resampling is conditional on the fact that the ancestor at index j is equal to i.
        Otherwise, it's the standard resampling
    Returns
    -------
    indices:
        Indices of the resampled particles.
    """
    N = weights.shape[0]
    indices = jax.random.choice(key, N, p=weights, shape=(N,), replace=True)
    if conditional:
        indices = indices.at[j].set(i)
    return indices


def killing(key: PRNGKey, weights: Array, i: int, j: int, conditional: bool = True) -> Array:
    """
    Conditional killing resampling. The weights are assumed to be normalised already.
    Compared to the multinomial resampling, this algorithm does not move the indices when the weights are uniform.

    Parameters
    ----------
    key:
        Random number generator key.
    weights:
        Weights of the particles.
    i, j:
        Conditional indices: the resampling is conditioned on the fact that the ancestor at index j is equal to i.
    conditional:
        If True, the resampling is conditional on the fact that the ancestor at index j is equal to i.
        Otherwise, it's the standard resampling

    Returns
    -------
    indices:
        Indices of the resampled particles.
    """

    # unconditional killing
    key_1, key_2, key_3 = jax.random.split(key, 3)

    N = weights.shape[0]
    w_max = weights.max()

    killed = (jax.random.uniform(key_1, (N,)) * w_max >= weights)
    idx = jnp.arange(N)
    idx = jnp.where(~killed, idx,
                    jax.random.choice(key_2, N, (N,), p=weights))
    if not conditional:
        return idx
    # Random permutation
    # TODO: logspace ?
    J_prob = (1. - weights / w_max) / N
    J_prob = J_prob.at[i].set(0.)
    J_prob_i = jnp.maximum(1 - jnp.sum(J_prob), 0.)
    J_prob = J_prob.at[i].set(J_prob_i)

    J = jax.random.choice(key_3, N, (), p=J_prob)
    idx = jnp.roll(idx, j - J)
    idx = idx.at[j].set(i)

    return idx


@partial(jax.jit, static_argnums=(1,))
def normalize(log_weights: Array, log_space=False) -> Array:
    """
    Normalize log weights to obtain unnormalized weights.

    Parameters
    ----------
    log_weights:
        Log weights to normalize.
    log_space:
        If True, the output is in log space. Otherwise, the output is in natural space.

    Returns
    -------
    log_weights/weights:
        Unnormalized weights.
    """
    log_weights -= logsumexp(log_weights)
    if log_space:
        return log_weights
    return jnp.exp(log_weights)
