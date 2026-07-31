import jax.numpy as jnp
from jax import Array

################################
#       helper functions       # 
################################

def unpack_params(params):
    trainable = params.get("trainable", {})
    fixed = params.get("fixed", {})
    return {**fixed, **trainable}

def ou_diag_transition(A, chol_Q, dt):
    """
    Exact OU transition for dz = -diag(A_diag) z dt + chol_Q dB_t.

    Parameters
    ----------
    A:       (D, D) Diagonal transition matrix
    chol_Q:  Cholesky factor of the covariance
    
    Returns
    -------
    F:   (D, D) transition matrix
    Cov: (D, D) transition covariance
    """
    Q = chol_Q @ chol_Q.T

    A_diag = jnp.diag(A)
    a_sum = A_diag[:, None] + A_diag[None, :]
    factor = jnp.where(
        jnp.abs(a_sum) > 1e-10,
        (1.0 - jnp.exp(-a_sum * dt)) / a_sum,
        dt,
    )

    F = jnp.diag(jnp.exp(-A_diag * dt))
    Cov = factor * Q
    chol_Cov = jnp.linalg.cholesky(Cov)
    return F, chol_Cov

def _diag_or_vector_at(chol_R: Array, i: Array):
    """
    Returns the scalar observation standard deviation for bond i.

    Parameters
    ----------
    chol_R: (dim,) or (dim, dim)
    """
    if chol_R.ndim == 1:
        return chol_R[i]
    elif chol_R.ndim == 2:
        return chol_R[i, i]
    else:
        raise ValueError("chol_R must have shape (dim,) or (dim, dim).")

def _logdiffexp(a: Array, b: Array):
    """ Computes log(exp(a) - exp(b)), assuming a >= b. """
    # return a + jnp.log1p(-jnp.exp(b - a))
    return a + jnp.log1p(-jnp.exp(jnp.minimum(b - a, -1e-7)))

