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
from jax.numpy.linalg import inv

from jax.scipy.stats import norm
from chex import PRNGKey, Array

from jax.scipy.linalg import solve_triangular

from rbsmc.utils.math import logdet, mvn_logpdf
from rbsmc.utils.inverse_gamma import inverse_gamma


################################
#      helper functions       # 
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


def _construct_cov_cholesky(beta, delta, jitter: float = 1e-6):
    """
    Constructs chol_Q from modified Cholesky precision parameters.

    Q^{-1} = T.T @ D^{-1} @ T
    T = I - tril(beta, -1)
    D = diag(exp(log_d))

    beta[j, k], j > k, is the triangular regression coefficient.
    """
    dim = beta.shape[-1]
    beta = jnp.tril(beta, -1)

    T = jnp.eye(dim, dtype=beta.dtype) - beta

    D_inv_sqrt_T = T / jnp.sqrt(delta[:, None])
    precision = D_inv_sqrt_T.T @ D_inv_sqrt_T

    Q = jnp.linalg.solve(precision, jnp.eye(dim, dtype=beta.dtype))
    Q = 0.5 * (Q + Q.T)

    return jnp.linalg.cholesky(Q + jitter * jnp.eye(dim, dtype=beta.dtype))


def sample_horseshoe_chol(
        key,
        dim,
        base_vol: float = 0.10,
        vol_slope: float = 0.40,
        tau_scale: float = 0.20,
        max_tau: float = 0.50,
        max_lambda: float = 10.0,
        max_beta: float = 0.50,
):
    """
    Samples a near-sparse eta innovation covariance from a horseshoe-style
    modified Cholesky parameterisation.

    No Bernoulli mask is used. The horseshoe gives continuous shrinkage:
    most beta values are small, but some can be meaningfully nonzero.
    """
    key_tau, key_lambda, key_beta = jr.split(key, 3)

    lower = jnp.tril(jnp.ones((dim, dim), dtype=bool), -1)

    tau_true = tau_scale * jnp.abs(jr.cauchy(key_tau, ()))
    tau_true = jnp.clip(tau_true, 1e-3, max_tau)

    llambda_true = jnp.abs(jr.cauchy(key_lambda, (dim, dim)))
    llambda_true = jnp.clip(llambda_true, 1e-3, max_lambda)
    llambda_true = jnp.where(lower, llambda_true, 1.0)

    beta_raw = tau_true * llambda_true * jr.normal(key_beta, (dim, dim))
    beta_true = jnp.where(lower, beta_raw, 0.0)
    beta_true = jnp.clip(beta_true, -max_beta, max_beta)
    beta_true = jnp.tril(beta_true, -1)

    vol_eta = base_vol * jnp.linspace(1.0, 1.0 + vol_slope, dim)
    delta_true = vol_eta**2

    chol_H_true = _construct_cov_cholesky(beta_true, delta_true)

    return chol_H_true, beta_true, delta_true, llambda_true, tau_true


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


def get_prior_params(key, D, T, steps, phi, log_var):
    
    # --- dynamics config ---
    A = phi * jnp.eye(D)
    CHOL_Q0 = 0.1 * jnp.eye(D)
    CHOL_H0 = 0.1 * jnp.eye(D)
    CHOL_Q = 10 ** (log_var / 2) * jnp.eye(D)  # independent spreads

    CHOL_H, BETA, DELTA, LLAMBDA, TAU = sample_horseshoe_chol(
        key,
        dim=D,
        base_vol=0.10,
        vol_slope=0.40,
        tau_scale=0.05,
    )

    CHOL_R = 0.1 * jnp.eye(D)
    PSI = 0.05 * jnp.ones(D)
    ALPHA = 0.10 * jnp.ones(D)
    DTs = jnp.repeat(T / steps, steps)

    return (A, CHOL_Q0, CHOL_H0, CHOL_Q, CHOL_H, 
            BETA, DELTA, LLAMBDA, TAU, CHOL_R, PSI, ALPHA, DTs)


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
    chol_H = _construct_cov_cholesky(params["beta"], params["delta"])

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
        A: Array,
        tau: float,
        chol_Q0: Array, 
        chol_H0: Array,
        chol_Q: Array,
        llambda: Array,
        chol_R: Array,
        psi: Array,
        alpha: Array,
        dts: Array,
    ):

    beta = jnp.zeros((dim, dim))
    delta = 0.1 * jnp.ones((dim,))

    trainable = {"beta": beta, "delta": delta}
    fixed = {
        "A": A,
        "chol_Q0": chol_Q0, "chol_Q": chol_Q,
        "chol_H0": chol_H0, 
        "chol_R": chol_R,
        "psi": psi, "alpha": alpha, 
        "lambda": llambda, "tau": tau,
        "dts": dts
    }

    return {"trainable": trainable, "fixed": fixed}


def dummy_init(params, data):
    params = unpack_params(params)
    chol_Q = params["chol_Q"]
    ys = data[0]
 
    D = chol_Q.shape[-1]
    B, T = ys.shape[:2]

    return jnp.zeros(shape=(B, T, D)), jnp.zeros(shape=(B, T, D))


def stabilise_params(params):
    return params
    # grad = params["trainable"]["grad"]

    # A = grad["A"]
    # A_diag = jnp.diag(A)
    # A_diag = jnp.clip(A_diag, 1e-3, 1.0)

    # beta = grad["beta"]
    # beta = jnp.clip(beta, -0.75, 0.75)

    # log_d = grad["log_d"]
    # log_d = jnp.clip(log_d, 2 * jnp.log(1e-2), 2 * jnp.log(0.5))

    # grad = {
    #     **grad,
    #     "A": jnp.diag(A_diag),
    #     "beta": beta,
    #     "log_d": log_d
    # }

    # return {
    #     **params,
    #     "trainable": {
    #         **params["trainable"],
    #         "grad": grad,
    #     },
    # }


#########################################
#       Parameter update function       #
#########################################


@jax.jit
def update(key, params, samples):
    """
    Gibbs update for the modified Cholesky parameters of the eta
    innovation covariance.

    Parameters
    ----------
    key:      JAX PRNG key.
    params:   dict of prior parameters
    samples:  Tuple where:
                - samples[0]: (B, T, D) is half-spread samples
                - samples[1]: (B, T, D) is mid-price samples

    Returns
    -------
    Updated parameter PyTree.
    """
    key_beta, key_delta = jr.split(key)

    p = unpack_params(params)
    llambda = p["lambda"]
    tau = p["tau"]
    beta = p["beta"]
    delta = p["delta"]
    dts = p["dts"]

    # Bayesian updates
    F = jnp.eye(samples[1].shape[-1])
    beta_next, rss, V = _update_beta(key_beta, samples=samples[1], tau=tau, local_scale=llambda, 
                                     delta=delta, dts=dts, F=F)
    delta_next = _update_delta(key_delta, samples=samples[1], beta=beta_next, rss=rss, local_var=V)
    
    return {
        **params,
        "trainable": {
            **params["trainable"],
            "beta": beta_next,
            "delta": delta_next,
        },
    }


def _update_beta(
        key: PRNGKey,
        samples: Array,
        tau,
        local_scale,
        delta,
        dts: Array,
        F: Array,
        jitter: float = 1e-6,
):
    B, T, D = samples.shape
    dtype = samples.dtype
    tiny = jnp.finfo(dtype).tiny

    lower = jnp.tril(jnp.ones((D, D), dtype=bool), -1)

    residuals = (
        samples[:, 1:, :] - samples[:, :-1, :]
    ) / jnp.sqrt(dts[1:, None])

    residuals = residuals.reshape(B * (T - 1), D)

    local_var = jnp.where(
        lower,
        tau**2 * jnp.square(local_scale),
        1.0,
    )
    local_var = jnp.maximum(local_var, tiny)

    prior_precision = jnp.where(
        lower,
        1.0 / local_var,
        1.0,
    )

    gram = residuals.T @ residuals
    active_outer = lower[:, :, None] & lower[:, None, :]

    # R_d.T R_d / delta_d + V_d^{-1}
    precision = jnp.where(
        active_outer,
        gram[None, :, :] / delta[:, None, None],
        0.0,
    )

    precision = precision + (
        jnp.eye(D, dtype=dtype)[None, :, :]
        * prior_precision[:, None, :]
    )

    precision = precision + jitter * jnp.eye(D, dtype=dtype)[None]

    # R_d.T r_d / delta_d
    rhs = jnp.where(
        lower,
        gram.T / delta[:, None],
        0.0,
    )

    chol_precision = jnp.linalg.cholesky(precision)

    beta_mean = jnp.linalg.solve(
        precision,
        rhs[..., None],
    )[..., 0]

    standard_normal = jr.normal(
        key,
        shape=(D, D),
        dtype=dtype,
    )

    posterior_noise = jnp.linalg.solve(
        jnp.swapaxes(chol_precision, -1, -2),
        standard_normal[..., None],
    )[..., 0]

    # No multiplication by sqrt(delta).
    beta_next = beta_mean + posterior_noise
    beta_next = jnp.where(lower, beta_next, 0.0)

    fitted = residuals @ beta_next.T
    regression_error = residuals - fitted
    rss = jnp.sum(jnp.square(regression_error), axis=0)

    return beta_next, rss, local_var


def _update_delta(
        key: PRNGKey,
        samples: Array,
        beta: Array,
        rss,
        local_var,
):
    B, T, D = samples.shape
    dtype = samples.dtype
    tiny = jnp.finfo(dtype).tiny

    n_transitions = B * (T - 1)

    concentration = jnp.full(
        (D,),
        0.5 * n_transitions,
        dtype=dtype,
    )

    scale = 0.5 * rss

    return inverse_gamma(
        key,
        concentration,
        jnp.maximum(scale, tiny),
    )

# def _update_beta(
#         key: PRNGKey,
#         samples: Array,
#         tau,
#         local_scale,
#         delta,
#         dts: Array,
#         F: Array,
#         jitter: float = 1e-6
#     ):
#     """
#     """

#     B, T, D = samples.shape
#     dtype = samples.dtype
#     tiny = jnp.finfo(dtype).tiny

#     # mask for lower triangular
#     lower = jnp.tril(jnp.ones((D, D), dtype=bool), -1)

#     # calculate residuals  # TODO add F for transition matrix
#     residuals = (samples[:, 1:, :] - samples[:, :-1, :] ) / jnp.sqrt(dts[1:, None])
#     residuals = residuals.reshape((B * (T - 1), D))   # flatten over batches

#     # calculate V_d = tau^2 diag(lambda[d, :d]^2)
#     local_var = jnp.where(lower, tau**2 * jnp.square(local_scale), 1.0)
#     local_var = jnp.maximum(local_var, tiny)

#     # calculate V_d^{-1}
#     prior_precision = jnp.where(lower, 1.0 / local_var, 1.0)

#     # calculate R_d.T @ R_t + V_d^{-1}
#     gram = residuals.T @ residuals
#     active_outer = lower[:, :, None] & lower[:, None, :]
#     precision = jnp.where(active_outer, gram[None, :, :], 0.0)
#     precision = precision + (jnp.eye(D)[None, :, :] * prior_precision[:, None, :])
#     precision = precision + jitter * jnp.eye(D)[None]  # numerical stabilisation

#     # calculate \hat beta = precision^{-1} @ R_d.T @ r_j
#     rhs = jnp.where(lower, gram.T, 0.0)
#     chol_precision = jnp.linalg.cholesky(precision)
#     beta_mean = jnp.linalg.solve(precision, rhs[..., None])[..., 0]

#     # perform regression
#     standard_normal = jr.normal(key, shape=(D, D))

#     posterior_noise = jnp.linalg.solve(jnp.swapaxes(chol_precision, -1, -2), standard_normal[..., None])[..., 0]
#     beta_next = beta_mean + jnp.sqrt(jnp.maximum(delta, tiny))[:, None] * posterior_noise
#     beta_next = jnp.where(lower, beta_next, 0.0)

#     fitted = residuals @ beta_next.T
#     regression_error = residuals - fitted
#     rss = jnp.sum(jnp.square(regression_error), axis=0)

#     return beta_next, rss, local_var


# def _update_delta(
#         key: PRNGKey,
#         samples: Array,
#         beta: Array,
#         rss,
#         local_var,

#     ):

#     B, T, D = samples.shape
#     dtype = samples.dtype
#     tiny = jnp.finfo(dtype).tiny

#     n_transitions = B * (T-1)

#     # mask for lower triangular
#     lower = jnp.tril(jnp.ones((D, D), dtype=bool), -1)

#     # beta_d.T V_d^{-1} beta_d
#     prior_quadratic = jnp.sum(jnp.where(lower, jnp.square(beta) / local_var, 0.0), axis=1)

#     # Row d has d active coefficients under zero-based indexing.
#     n_coefficients = jnp.arange(D, )
#     concentration = 0.5 * (n_transitions + n_coefficients)
#     scale = 0.5 * (rss + prior_quadratic)
#     delta_next = inverse_gamma(key, concentration, jnp.maximum(scale, tiny))
#     delta_next = jnp.maximum(delta_next, jnp.finfo(delta_next.dtype).tiny)

#     return delta_next