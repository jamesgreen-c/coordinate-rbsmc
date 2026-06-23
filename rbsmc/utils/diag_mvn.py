import math
from functools import partial
from jax import numpy as jnp


def diag_mvn_logpdf(x, m, sigma, constant=True):
    """
    Computes the log of the probability density function of a diagonal multivariate normal distribution with mean m and
    stdev vector sigma.
    This saves memory in the large D case where forming the covariance matrix is too costly

    Parameters
    ----------
    x: Array
        Point where the density is evaluated.
    m: Array
        Mean of the distribution.
    sigma: Array
        Vector of standard deviations.
    constant: bool, optional
        Whether to return the log density with respect to the Lebesgue measure or with respect to the Gaussian measure.

    Returns
    -------
    logpdf: float
        Log of the probability density function evaluated at x.
    """
    # Numerically ignore nans and infs

    out = _diag_logpdf(x, m, sigma)
    if constant:
        normalizing_constant = _get_constant(sigma)
    else:
        normalizing_constant = 0.

    return out - normalizing_constant


@partial(jnp.vectorize, signature="(n)->()")
def _get_constant(sigma):
    valid = jnp.isfinite(sigma) & (sigma > 0.0)
    dim = jnp.sum(valid)
    logdet = jnp.sum(jnp.where(valid, jnp.log(sigma), 0.0))
    return logdet + 0.5 * dim * math.log(2.0 * math.pi)


@partial(jnp.vectorize, signature="(n),(n),(n)->()")
def _diag_logpdf(x, m, sigma):
    valid = jnp.isfinite(sigma) & (sigma > 0.0)
    z = jnp.where(valid, (x - m) / sigma, 0.0)
    return -0.5 * jnp.sum(z * z)




