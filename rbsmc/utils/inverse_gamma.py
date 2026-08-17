import jax.numpy as jnp
import jax.random as jr


# def inverse_gamma(key, concentration, scale):
#     """
#     Sample InvGamma(concentration, scale), with density

#         p(x) ∝ x^(-concentration - 1) exp(-scale / x).
#     """
#     dtype = jnp.result_type(concentration, scale)
#     concentration = jnp.asarray(concentration, dtype=dtype)
#     scale = jnp.asarray(scale, dtype=dtype)

#     tiny = jnp.finfo(dtype).tiny
#     gamma_draw = jr.gamma(key, concentration, dtype=dtype)

#     return scale / jnp.maximum(gamma_draw, tiny)


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
