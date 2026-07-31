from functools import partial

import jax
import jax.numpy as jnp

from jax import Array
from jax.scipy.linalg import solve_triangular
from jax.scipy.stats import norm

from rbsmc.utils.mvn import mvn_logpdf

from bayesian_rework.utils import (ou_diag_transition, 
                                   unpack_params, 
                                   _diag_or_vector_at, 
                                   _logdiffexp)


####################################
#       corporate bond prior       #
####################################

def log_p0(params: dict, x0, constant: bool = True):
    """ Implement the t=0 logpdf for states """
    z0, eta0 = x0
    dim = z0.shape[-1]

    # extract params
    params = unpack_params(params)
    chol_Q0 = params["chol_Q0"]
    chol_H0 = params["chol_H0"]

    # compute inverse cholesky factors
    inv_chol_Q0 = solve_triangular(chol_Q0, jnp.eye(dim), lower=True)
    inv_chol_H0 = solve_triangular(chol_H0, jnp.eye(dim), lower=True)

    @partial(jnp.vectorize, signature=("(n),(n)->()"))
    def _logpdf(_z0, _eta0):
        val = mvn_logpdf(_z0, jnp.zeros_like(_z0), None, chol_inv=inv_chol_Q0, constant=constant)
        val += mvn_logpdf(_eta0, jnp.zeros_like(_eta0), None, chol_inv=inv_chol_H0, constant=constant)
        return val
    
    return _logpdf(z0, eta0)

def log_pt(params: dict, xp, x, dt, constant: bool = True):
    """ Implement the prior transition logpdf for states """
    zp, etap = xp
    z, eta = x
    dim = z.shape[-1]

    # extract params
    params = unpack_params(params)
    A = params["A"]
    chol_Q = params["chol_Q"]
    chol_H = params["chol_H"]

    # calculate exact transition dynamics
    Ft, chol_Qt = ou_diag_transition(A, chol_Q, dt)
    chol_Ht = jnp.sqrt(dt) * chol_H

    # compute inverse cholesky factors
    inv_chol_Qt = solve_triangular(chol_Qt, jnp.eye(dim), lower=True)
    inv_chol_Ht = solve_triangular(chol_Ht, jnp.eye(dim), lower=True)

    @partial(jnp.vectorize, signature=("(n),(n),(n),(n)->()"))
    def _logpdf(_zp, _etap, _z, _eta):
        val = mvn_logpdf(_z, _zp @ Ft.T, None, chol_inv=inv_chol_Qt, constant=constant)
        val += mvn_logpdf(_eta, _etap, None, chol_inv=inv_chol_Ht, constant=constant)
        return val

    return _logpdf(zp, etap, z, eta)


def log_ht(params, x, data: tuple[Array]):
    """
    Corporate-bond event log-likelihood.

    Parameters
    ----------
    i:         (,) Jax integer the relevant bond idx
    y:         (,) Value of the observed trade
    obs_type:  (,) Jax integer in [0, 1, 2, 3, 4] to identify the type of trade observed
    zs:        (dim,) Jax Array of the sampled log half-spreads
    etas:      (dim,) Jax Array of the sampled mid-YtBs
    alpha:     (dim,) Jax float for alpha (D2D half-width) for the relevant bond index
    psi:       (dim,) Baseline half-spread scale
    chol_R:    (dim,) or (dim, dim) Observation noise standard deviations

    Returns
    -------
    val: Scalar log-likelihood contribution.
    """
    y, i, obs_type = data
    zs, etas = x

    # extract params
    params = unpack_params(params)
    alpha = params["alpha"]
    psi = params["psi"]
    chol_R = params["chol_R"]

    # extract relevant bond dimd
    z_i = zs[..., i]
    eta_i = etas[..., i]
    alpha_i = alpha[i]

    # retrieve bond-specific emission parameters
    r_i = _diag_or_vector_at(chol_R, i)
    spread_i = psi[i] * jnp.exp(z_i)

    case_0 = lambda: norm.logpdf(y, loc=eta_i - spread_i, scale=r_i)          # D2C buy: Y = eta_i - psi_i + eps
    case_1 = lambda: norm.logpdf(y, loc=eta_i + spread_i, scale=r_i)          # D2C sell: Y = eta_i + psi_i + eps
    case_2 = lambda: norm.logcdf((eta_i - spread_i) - y, loc=0.0, scale=r_i)  # traded-away buy RFQ:  observed quote Z, condition eta_i - psi_i + eps >= Z
    case_3 = lambda: norm.logcdf(y - (eta_i + spread_i), loc=0.0, scale=r_i)  # traded-away sell RFQ: observed quote Z, condition eta_i + psi_i + eps <= Z

    def case_4():
        # D2D: observed Y, condition Y in [eta_i - alpha_i + eps, eta_i + alpha_i + eps]
        lo = y - eta_i - alpha_i
        hi = y - eta_i + alpha_i
        log_hi = norm.logcdf(hi, loc=0.0, scale=r_i)
        log_lo = norm.logcdf(lo, loc=0.0, scale=r_i)
        val = _logdiffexp(log_hi, log_lo)

        # finite floor for log probs stop infinite loss
        tiny = jnp.log(jnp.finfo(val.dtype).tiny)
        return jnp.maximum(val, tiny)
    
    return jax.lax.switch(obs_type, [case_0, case_1, case_2, case_3, case_4])

