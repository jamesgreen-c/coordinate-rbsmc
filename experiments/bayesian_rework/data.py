from functools import partial 

import jax
import jax.numpy as jnp
import jax.random as jr

from jax import Array
from jax.random import PRNGKey

from bayesian_rework.utils import (ou_diag_transition, _diag_or_vector_at)

from rbsmc.utils.iw import InverseWhishart

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

    done_buy = eta[i] - spread_i + eps
    done_sell = eta[i] + spread_i + eps

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
        sparsity_factor: float = 1.0,
        **kwargs
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

    init_key, event_key, sampling_key = jr.split(key, 3)
    K = dts.shape[0]

    init_key_z, init_key_eta = jr.split(init_key)

    z0 = chol_Q0 @ jr.normal(init_key_z, (dim,))
    eta0 = chol_H0 @ jr.normal(init_key_eta, (dim,))

    key_bond, key_type, key_y = jr.split(event_key, 3)

    bond_weights = jnp.ones((dim,))
    bond_weights = bond_weights.at[:-1].set(sparsity_factor)
    bond_probs = bond_weights / jnp.sum(bond_weights)
    bond_idxs = jr.categorical(key_bond, jnp.log(bond_probs), shape=(K,)).astype(jnp.int32)

    event_types = jr.randint(key_type, (K,), minval=0, maxval=5)
    keys_y = jr.split(key_y, K)

    Fs, chol_Bs = jax.vmap(lambda dt: ou_diag_transition(A, chol_Q, dt))(dts)
    eps_zs, eps_etas = jr.normal(sampling_key, (2, K, dim))

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
    
    LLAMBDA = jnp.eye(D)
    H = InverseWhishart.sample(key, D, LLAMBDA)
    CHOL_H = jnp.linalg.cholesky(H)

    CHOL_R = 0.1 * jnp.eye(D)
    PSI = 0.05 * jnp.ones(D)
    ALPHA = 0.10 * jnp.ones(D)
    DTs = jnp.repeat(T / steps, steps)

    params = {
        "A": A,
        "CHOL_Q0": CHOL_Q0,
        "CHOL_H0": CHOL_H0,
        "CHOL_Q":  CHOL_Q,
        "LLAMBDA": LLAMBDA,
        "H": H,
        "CHOL_H": CHOL_H,
        "CHOL_R": CHOL_R,
        "PSI": PSI,
        "ALPHA": ALPHA,
    }
    return params, DTs