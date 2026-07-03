"""
Continuous-time corporate bond mid-YtB and spread model.

Latent state:
    s_t = (u_t, z_t)

where
    u_t: mid-YtB process
    z_t: log half-spread process
    psi_t = Psi * exp(z_t)

Observation encoding:
    obs = [value, bond_idx, event_type, alpha]

event_type:
    0: client buys from dealer D       Y = u_i - psi_i + eps
    1: client sells to dealer D        Y = u_i + psi_i + eps
    2: traded-away client buy RFQ      observed Z, with u_i - psi_i + eps >= Z
    3: traded-away client sell RFQ     observed Z, with u_i + psi_i + eps <= Z
    4: D2D trade                       observed Y in [u_i - alpha_i + eps, u_i + alpha_i + eps]
"""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import norm
from chex import PRNGKey, Array

from jax.scipy.linalg import solve_triangular

from rbsmc.utils.math import logdet, mvn_logpdf

################################
#      helper functions       # 
################################

def unpack_params(params):
    trainable = params.get("trainable", {})
    grad = trainable.get("grad", {})
    exact = trainable.get("exact", {})
    fixed = params.get("fixed", {})
    return {**fixed, **grad, **exact}


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


##############################
#       data functions       # 
##############################

def emission(
        key: PRNGKey,
        z: Array,
        eta: Array,
        psi: Array,
        chol_R: Array,
        alpha: Array,
        bond_idx: Array,
        event_type: Array,
):
    """
    Simulates one corporate-bond event observation.

    Parameters
    ----------
    key:        PRNGKey
    eta:        (dim,) Mid-YtB state
    z:          (dim,) Log half-spread state
    psi:        (dim,) Baseline half-spread scale Psi
    chol_R:     (dim,) or (dim, dim) Observation noise standard deviations
    alpha:      (dim,) D2D interval half-widths
    bond_idx:   Integer bond index
    event_type: Integer event type in {0, 1, 2, 3, 4}

    Returns
    -------
    obs_value: Scalar observation value.

    Notes
    -----
    For event types 2 and 3, obs_value is the dealer quote Z.
    """

    key_eps, key_aux = jax.random.split(key)

    i = bond_idx
    r = _diag_or_vector_at(chol_R, i)

    spread_i = psi[i] * jnp.exp(z[i])
    eps = r * jax.random.normal(key_eps)

    done_buy = eta[i] - spread_i + eps
    done_sell = eta[i] + spread_i + eps

    # for traded-away events we simulate a quote Z consistent with the event.
    margin = jnp.abs(r * jax.random.normal(key_aux))

    # observation cases
    case_0 = lambda: done_buy                                                # client buys from dealer D
    case_1 = lambda: done_sell                                               # client sells to dealer D
    case_2 = lambda: done_buy - margin                                       # client buys from another dealer
    case_3 = lambda: done_sell + margin                                      # client sells to another dealer
    case_4 = lambda: eta[i] + eps + jr.uniform(key_aux,
                                               shape=(),
                                               minval=-alpha[i],
                                               maxval= alpha[i])             # D2D trade: observed Y lies inside an interval around u_i + eps

    return jax.lax.switch(event_type, [case_0, case_1, case_2, case_3, case_4])


# dynamics functions
def get_half_spread_dynamics(A: Array, chol_Q_z: Array):
    def drift(t, z):
        return -A @ z
    def diffusion(t, z):
        return chol_Q_z
    return drift, diffusion


def get_ytb_dynamics(chol_Q_eta: Array):    
    def drift(t, eta):
        return jnp.zeros_like(eta)
    def diffusion(t, eta):
        return chol_Q_eta
    return drift, diffusion


@partial(jax.jit, static_argnums=(1, 11,))
def get_data(
        key: PRNGKey,
        dim: int,
        dts: Array,
        A: Array,
        psi: Array,
        chol_Q0: Array,
        chol_Q: Array,
        chol_H0: Array,
        chol_H: Array,
        chol_R: Array,
        alpha: Array,
        sparsity_factor: float = 10.0,
):
    """
    Simulates corporate-bond latent states and sparse event observations.

    Parameters
    ----------
    key:             PRNGKey
    dim:             Number of bonds
    dts:             (K,) Time increments
    A:               (dim, dim) Discrete-time transition matrix for z
    psi:             (dim,) Baseline half-spread scale
    chol_Q0:         (dim, dim) Initial Cholesky factor for z_0
    chol_Q:          (dim, dim) Transition Cholesky factor for z
    chol_H0:         (dim, dim) Initial Cholesky factor for eta_0
    chol_H:          (dim, dim) Transition Cholesky factor for eta
    chol_R:          (dim,) or (dim, dim) Observation noise standard deviations
    alpha:           (dim,) D2D interval half-widths
    sparsity_factor: Observation frequency ratio between non-final bonds and final bond.

    Returns
    -------
    xs:   Tuple (zs, etas)
            zs:   (K, dim)
            etas: (K, dim)

    obs:  Tuple (bond_idxs, event_types, alphas, obs_values)
    """

    init_key, event_key, sampling_key = jax.random.split(key, 3)
    K = dts.shape[0]

    init_key_z, init_key_eta = jax.random.split(init_key)

    z0 = chol_Q0 @ jax.random.normal(init_key_z, (dim,))
    eta0 = chol_H0 @ jax.random.normal(init_key_eta, (dim,))

    key_bond, key_type, key_y = jax.random.split(event_key, 3)

    bond_weights = jnp.ones((dim,))
    bond_weights = bond_weights.at[:-1].set(sparsity_factor)
    bond_probs = bond_weights / jnp.sum(bond_weights)
    bond_idxs = jax.random.categorical(key_bond, jnp.log(bond_probs), shape=(K,)).astype(jnp.int32)

    event_types = jax.random.randint(key_type, (K,), minval=0, maxval=5)
    keys_y = jax.random.split(key_y, K)

    Fs, chol_Bs = jax.vmap(lambda dt: ou_diag_transition(A, chol_Q, dt))(dts)
    eps_zs, eps_etas = jax.random.normal(sampling_key, (2, K, dim))

    def body(carry, inps):
        z_k, eta_k = carry
        dt, F, chol_B, eps_z, eps_eta, key_y_k, bond_idx, event_type = inps

        # sample next latent state
        z_kp1 = z_k @ F.T + eps_z @ chol_B.T
        eta_kp1 = eta_k + jnp.sqrt(dt) * (eps_eta @ chol_H.T)
        x_kp1 = (z_kp1, eta_kp1)

        # sample observation
        obs_value = emission(key_y_k, z_kp1, eta_kp1, psi, chol_R, alpha, bond_idx, event_type)
        obs_k = (obs_value, bond_idx, event_type)

        return x_kp1, (x_kp1, obs_k)

    carry0 = (z0, eta0)
    inps = (dts, Fs, chol_Bs, eps_zs, eps_etas, keys_y, bond_idxs, event_types)
    _, (xs, obs) = jax.lax.scan(body, carry0, inps)
    return xs, obs


#################################
#          prior model          # 
#################################

def log_p0(params, x0, constant: bool = True):
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

def log_pt(params, xp, x, dt, constant: bool = True):
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


def log_potential(
        params,
        xs,
        data,
):
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
    zs, etas = xs

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



#################################################
#       Parameter initialisation function       # 
#################################################

def init(
        dim, 
        phi: float,
        chol_Q0: Array, 
        chol_H0: Array,
        chol_Q: Array, 
        chol_H: Array,
        chol_R: Array,
        psi: Array,
        alpha: Array,
        stationary: bool = False
    ):

    A = phi * jnp.eye(dim)

    if stationary:
        grad = {"A": A}
        exact = {}
        fixed = {
            "chol_Q0": chol_Q0, "chol_Q": chol_Q,
            "chol_H0": chol_H0, "chol_H": chol_H,
            "chol_R": chol_R,
            "psi": psi, "alpha": alpha
        }

    else:
        grad = {
            "A": A,
            "chol_Q0": chol_Q0, "chol_Q": chol_Q,
            "chol_H0": chol_H0, "chol_H": chol_H,
            "chol_R": chol_R
        }
        exact = {}
        fixed = {"psi": psi, "alpha": alpha}
    
    trainable = {"grad": grad, "exact": exact}
    return {"trainable": trainable, "fixed": fixed}


def sample_init(key, params, data):
    params = unpack_params(params)
    chol_Q = params["chol_Q"]
    ys = data[0]
 
    D = chol_Q.shape[-1]
    B, T = ys.shape[:2]

    # Not problematic as no conditional logic currently
    return jnp.zeros(shape=(B, T, D)), jnp.zeros(shape=(B, T, D))

def stabilise_params(params):
    grad = params["trainable"]["grad"]

    A = grad["A"]
    A_diag = jnp.diag(A)
    A_diag = jnp.clip(A_diag, 1e-3, 1.0)

    grad = {
        **grad,
        "A": jnp.diag(A_diag),
    }

    return {
        **params,
        "trainable": {
            **params["trainable"],
            "grad": grad,
        },
    }