from functools import partial 

import jax
import jax.numpy as jnp
import jax.random as jr

from jax import Array
from jax.random import PRNGKey
from jax.tree_util import tree_map

from experiments.bayesian.utils import (ou_diag_transition, _diag_or_vector_at)
from experiments.bayesian.dataset import CorporateBondDataset

from rbsmc.utils.iw import InverseWhishart
from rbsmc.utils.inverse_gamma import inverse_gamma

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

    key_eps, key_aux = jr.split(key)

    i = bond_idx
    r = _diag_or_vector_at(chol_R, i)

    spread_i = psi[i] * jnp.exp(z[i])
    eps = r * jr.normal(key_eps)
    # print("spread :", spread_i)
    # print("eps: ", eps)

    done_buy = eta[i] - spread_i + eps
    done_sell = eta[i] + spread_i + eps

    # print("done buy: ", done_buy)

    # for traded-away events we simulate a quote Z consistent with the event.
    margin = jnp.abs(r * jr.normal(key_aux))

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


def get_data(
        key: PRNGKey,
        dim: int,
        dts: Array,
        params: dict,
        # m0: Array,
        # A: Array,
        # psi: Array,
        # Q0: Array,
        # Q: Array,
        # H0: Array,
        # H: Array,
        # R: Array,
        # alpha: Array,
        sparsity_factor: float = 1.0,
        **kwargs
) -> CorporateBondDataset:
    """
    Simulates corporate-bond latent states and sparse event observations.

    Parameters
    ----------
    key:             PRNGKey
    dim:             Number of bonds
    dts:             (K-1,) Time increments
    A:               (dim, dim) Discrete-time transition matrix for z
    psi:             (dim,) Baseline half-spread scale
    Q0:              (dim, dim) Initial covariance for z_0
    Q:               (dim, dim) Transition covariance for z
    H0:              (dim, dim) Initial covariance for eta_0
    H:               (dim, dim) Transition covariance for eta
    R:               (dim,) or (dim, dim) Observation noise standard deviations
    alpha:           (dim,) D2D interval half-widths
    sparsity_factor: Observation frequency ratio between non-final bonds and final bond.

    Returns
    -------
    xs:   Tuple (zs, etas)
            zs:   (K, dim)
            etas: (K, dim)

    obs:  Tuple (bond_idxs, event_types, alphas, obs_values)
    """

    A = params["A"]
    Q0 = params["Q0"]
    Q = params["Q"]
    M0 = params["m0"]
    H0 = params["H0"]
    H = params["H"]
    R = params["R"]
    PSI = params["psi"]
    ALPHA = params["alpha"]

    K = dts.shape[0] + 1
    init_key, event_key, sampling_key = jr.split(key, 3)
    init_key_z, init_key_eta = jr.split(init_key)
    key_bond, key_type, key_y = jr.split(event_key, 3)

    # precompute cholesky's
    chol_Q0 = jnp.linalg.cholesky(Q0)
    chol_H0 = jnp.linalg.cholesky(H0)
    chol_Q = jnp.linalg.cholesky(Q)
    chol_H = jnp.linalg.cholesky(H)
    chol_R = jnp.linalg.cholesky(R)

    # presample observation quantities
    bond_weights = jnp.ones((dim,))
    bond_weights = bond_weights.at[:-1].set(sparsity_factor)
    bond_probs = bond_weights / jnp.sum(bond_weights)
    bond_idxs = jr.categorical(key_bond, jnp.log(bond_probs), shape=(K,)).astype(jnp.int32)
    event_types = jr.randint(key_type, (K,), minval=0, maxval=5)

    keys_y = jr.split(key_y, K)

    # t=0 data
    z0 = chol_Q0 @ jr.normal(init_key_z, (dim,))
    eta0 = M0 + chol_H0 @ jr.normal(init_key_eta, (dim,))
    obs_val0 = emission(keys_y[0], z0, eta0, PSI, chol_R, ALPHA, bond_idxs[0], event_types[0])
    obs0 = (obs_val0, bond_idxs[0], event_types[0])

    # t=1,...,T data
    Fs, chol_Bs = jax.vmap(lambda dt: ou_diag_transition(A, Q, dt))(dts)
    eps_zs, eps_etas = jr.normal(sampling_key, (2, K-1, dim))

    def body(carry, inps):
        z_k, eta_k = carry
        dt, F, chol_B, eps_z, eps_eta, key_y_k, bond_idx, event_type = inps

        # sample next latent state
        z_kp1 = z_k @ F.T + eps_z @ chol_B.T
        eta_kp1 = eta_k + jnp.sqrt(dt) * (eps_eta @ chol_H.T)
        x_kp1 = (z_kp1, eta_kp1)

        # sample observation
        obs_value = emission(key_y_k, z_kp1, eta_kp1, PSI, chol_R, ALPHA, bond_idx, event_type)
        obs_kp1 = (obs_value, bond_idx, event_type)

        return x_kp1, (x_kp1, obs_kp1)

    x0 = (z0, eta0)
    inps = (dts, Fs, chol_Bs, eps_zs, eps_etas, keys_y[1:], bond_idxs[1:], event_types[1:])
    _, (xs, obs) = jax.lax.scan(body, x0, inps)

    xs = tree_map(lambda _x0, _x: jnp.concatenate((_x0[None], _x), axis=0), x0, xs)
    obs = tree_map(lambda _y0, _y: jnp.concatenate((_y0[None], _y), axis=0), obs0, obs)

    return CorporateBondDataset(
        D=dim, 
        dts=dts,
        data=obs,
        states=xs,
        params=params,
    )
    # return xs, obs


# def get_prior_params(key, D, T, steps, phi, log_var):
#     m0_key, H_key = jr.split(key)

#     # log half-spread transition matrix
#     A = phi * jnp.eye(D)

#     # mid-YtB initial mean, in percentage-point units
#     scale = 100
#     M0 = scale * jr.uniform(m0_key, shape=(D,), minval=0.5, maxval=1.0)

#     # covariance parameters
#     Q0 = 0.01 * jnp.eye(D)                             # initial uncertainty about log half-spreads
#     Q = 0.01 * jnp.eye(D)                              # daily log half-spread diffusion covariance
#     H0 = (scale * 0.01)**2 * jnp.eye(D)                # initial uncertainty about the mid-YtB
#     H = scale**2 * block_sparse_covariance(H_key, D)   # daily mid-YtB diffusion covariance
#     R = (scale * 0.000025)**2 * jnp.eye(D)             # observation-noise standard deviation approximately 0.2–0.3 bp

#     PSI = scale * 0.007 * jnp.ones(D)                  # baseline half-spread: approximately 0.5–0.8 bp
#     ALPHA = scale * 0.005 * jnp.ones(D)                # D2D interval half-width; example value of 0.5 bp

#     DTs = jnp.repeat(T / steps, steps)

#     params = {
#         "A": A,
#         "m0": M0,
#         "Q0": Q0,
#         "H0": H0,
#         "Q": Q,
#         "H": H,
#         "R": R,
#         "psi": PSI,
#         "alpha": ALPHA,
#     }

#     return params, DTs

def get_model_params(key, D, T, steps, phi):
    m0_key, H_key, H0_key = jr.split(key, 3)
    scale = 100

    # log half-spread transition matrix
    A = phi * jnp.eye(D)

    # mid-YtB initial mean, in percentage-point units
    MEAN_M0 = scale * 0.75 * jnp.ones(D)
    COV_M0 = (scale * 0.1)**2 * jnp.eye(D)
    M0 = jr.multivariate_normal(m0_key, MEAN_M0, COV_M0)
    # M0 = scale * jr.uniform(m0_key, shape=(D,), minval=0.5, maxval=1.0)

    # covariance parameters
    Q0 = 0.1 * jnp.eye(D)                              # initial uncertainty about log half-spreads
    Q = 0.1 * jnp.eye(D)                               # daily log half-spread diffusion covariance

    H0_SCALE = scale * 0.01 * jnp.ones(D)
    CONCENTRATION = 2 * jnp.ones(D)
    H0 = jax.vmap(lambda _k, _c, _s: inverse_gamma(_k, _c, _s))(jr.split(H0_key, D), CONCENTRATION, H0_SCALE)
    H0 = jnp.diag(H0)
    # H0 = (scale * 0.01)**2 * jnp.eye(D)               # initial uncertainty about the mid-YtB
    R = (scale * 0.00025)**2 * jnp.eye(D)               # observation-noise standard deviation approximately 0.2–0.3 bp

    if D == 3:
        # Guéant and Pu: volatilities in bp per sqrt(day)
        sigmas_bps = jnp.array([0.50, 0.62, 0.69])
        correlation = jnp.array([
            [1.000, 0.843, 0.835],
            [0.843, 1.000, 0.887],
            [0.835, 0.887, 1.000],
        ])

        # convert basis points into scaled percentage-point units
        sigmas = sigmas_bps / 100
        H = scale**2 * correlation * jnp.outer(sigmas, sigmas)
    else:
        H = scale**2 * block_sparse_covariance(H_key, D)

    PSI = scale * 0.007 * jnp.ones(D)                  # baseline half-spread: approximately 0.5–0.8 bp
    ALPHA = scale * 0.005 * jnp.ones(D)                # D2D interval half-width; example value of 0.5 bp
    DTs = jnp.repeat(T / steps, steps)

    params = {
        "A": A,
        "m0": M0,
        "Q0": Q0,
        "H0": H0,
        "Q": Q,
        "H": H,
        "R": R,
        "psi": PSI,
        "alpha": ALPHA,
        # "mean_m0": MEAN_M0,
        # "cov_m0": COV_M0,
        # "scale": H0_SCALE,
        # "concentration": CONCENTRATION,
    }

    return params, DTs

def block_sparse_covariance(key, D: int):

    key_permutation, key_signs, key_scale = jr.split(key, 3)

    rho = 0.8
    n_correlated_pairs = D // 3

    permutation = jr.permutation(key_permutation, D)

    signs = 2.0 * jr.bernoulli(
        key_signs,
        p=0.5,
        shape=(n_correlated_pairs,),
    ).astype(jnp.float64) - 1.0

    correlation = jnp.eye(D, dtype=jnp.float64)

    for pair in range(n_correlated_pairs):
        i = permutation[2 * pair]
        j = permutation[2 * pair + 1]

        pair_correlation = signs[pair] * rho
        correlation = correlation.at[i, j].set(pair_correlation)
        correlation = correlation.at[j, i].set(pair_correlation)

    # approximately 0.50–0.69 bp per sqrt(day), expressed
    # in percentage-point units
    sigmas = jr.uniform(
        key_scale,
        shape=(D,),
        minval=0.0050,
        maxval=0.0069,
        dtype=jnp.float64,
    )

    H = correlation * jnp.outer(sigmas, sigmas)

    return H