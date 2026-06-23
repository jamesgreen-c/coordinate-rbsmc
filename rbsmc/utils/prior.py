from typing import Callable, Union, Any

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey


def sample(
        key: PRNGKey,
        M_0: tuple[Callable, Callable], 
        M_t: Union[tuple[Callable, Callable], tuple[Callable, Callable, Any]],
        N: int,
        T: int,
        get_samples: bool = False
    ):
    """
    Sample N particles from the prior with no resampling.

    Parameters
    ----------
    key:
        Random number generator key.
    M_0:
        Sampler for the initial distribution.
    M_t:
        Sampler for the proposal distribution at time t.
         The first element is the sampling function, 
          the second element is the parameters as arguments.
    N:
        Number of particles to use.

    Returns
    -------

    """
    # housekeeping
    key_init, key_loop = jax.random.split(key, 2)
    M_t_rvs, _, prop_params = M_t if len(M_t) == 3 else (*M_t, None)
    M_0_rvs, _ = M_0

    # init
    x0 = M_0_rvs(key_init, N)

    # prior forward pass
    def body(x_t_m_1, inp):
        M_t_params, key_t = inp
        x_t = M_t_rvs(key_t, x_t_m_1, M_t_params)
        return x_t, x_t

    keys_loop = jax.random.split(key_loop, T-1)
    
    # sample from prior
    inputs = prop_params, keys_loop
    x_t, xs = jax.lax.scan(body, x0, inputs)
    
    if get_samples:
        xs = jnp.insert(xs, 0, x0, axis=0)
        return xs
    return x_t