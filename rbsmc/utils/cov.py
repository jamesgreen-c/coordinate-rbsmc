import jax.numpy as jnp


def wiener_covariance(dt, mesh):
    """
    Discretised Wiener covariance matrix on [0, dt].

    C[i, j] = min(t_i, t_j)
    C = dt * [[ 0 0 0 0 ... ]]
             [[ 0 1 1 1 ... ]]
             [[ 0 1 2 2 ... ]]
             [[ ...     ... ]] 

    Shape: (mesh, mesh)
    """
    ts = jnp.linspace(0.0, dt, mesh)
    return jnp.minimum(ts[:, None], ts[None, :])
