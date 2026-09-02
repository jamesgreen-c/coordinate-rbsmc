import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import gammaln


def inverse_gamma(key, concentration, scale):
    """
    Sample InvGamma(concentration, scale), with density

        p(x) ∝ x^(-concentration - 1) exp(-scale / x).
    """
    dtype = jnp.result_type(concentration, scale, jnp.float64)
    concentration = jnp.asarray(concentration, dtype=dtype)
    scale = jnp.asarray(scale, dtype=dtype)

    gamma_draw = jr.gamma(key, concentration, dtype=dtype)

    return scale / gamma_draw


def logpdf(x, concentration, scale):
    """
    Elementwise log-density of InvGamma(concentration, scale).
    """
    dtype = jnp.result_type(x, concentration, scale, jnp.float64)
    x = jnp.asarray(x, dtype=dtype)
    concentration = jnp.asarray(concentration, dtype=dtype)
    scale = jnp.asarray(scale, dtype=dtype)

    logpdf = (
        concentration * jnp.log(scale)
        - gammaln(concentration)
        - (concentration + 1) * jnp.log(x)
        - scale / x
    )

    return jnp.where(x > 0, logpdf, -jnp.inf)