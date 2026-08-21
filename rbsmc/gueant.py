"""
Gueant-style guided cSMC kernel for the corporate-bond model.
"""

from typing import Callable, Union, Any

import jax
from jax import Array
from jax.random import PRNGKey

import jax.random as jr
import jax.numpy as jnp

from jax.tree_util import tree_map
from jax.scipy.stats import norm
from jax.scipy.linalg import solve_triangular

from rbsmc.csmc import backward_sampling_pass, backward_scanning_pass
from rbsmc.utils.resamplings import normalize
from rbsmc.utils.mvn import mvn_logpdf


def kernel(
        key: PRNGKey,
        x_star,
        b_star: Array,
        M_0: tuple[Callable, Callable],
        Gamma_0: Callable,
        M_t_params,
        Gamma_t: Union[Callable, tuple[Callable, Any]],
        chol_Q_eta: Array,
        chol_R: Array,
        psi: Array,
        resampling_func: Callable,
        ancestor_move_func: Callable,
        N: int,
        backward: bool = False,
        conditional: bool = False,
):
    """
    Guéant guided cSMC kernel.

    Parameters
    ----------
    key:                 Random number generator key.
    x_star:              Reference trajectory to update.
    b_star:              Indices of the reference trajectory.
    M_0:                 Sampler for the initial distribution. 
    Gamma_0:             Initial weight function.
    M_t_params:          Params for the proposal distribution at time t.
    Gamma_t:             If a tuple, the first element is the function and the second element is the parameters.
    chol_Q_eta:
    chol_R:
    psi:
    resampling_func:     Resampling scheme to use.
    ancestor_move_func:  Function to move the last ancestor indices.
    N:                   Number of particles to use (N+1, if we include the reference trajectory).
    backward:            Whether to run the backward sampling kernel.
    conditional:         Whether to do conditional SMC or just SMC.
    """
    ###############################
    #        HOUSEKEEPING         #
    ###############################
    z_star, eta_star = x_star
    T, D = z_star.shape

    keys = jr.split(key, T + 1)
    key_init = keys[0]
    key_backward = keys[1]
    keys_forward = keys[2:]        # length T - 1

    Gamma_t, Gamma_params = Gamma_t if isinstance(Gamma_t, tuple) else (Gamma_t, None)
    M_0_rvs, M_0_logpdf = M_0

    ###########################################
    #       Guided proposal functions         #
    ###########################################

    def z_transition_rvs(key, x_t_m_1, params):
        # unpack
        _, F_t, chol_B_t, _ = params
        z_t_m_1, _ = x_t_m_1

        # propose only the half-spread state first, as in Guéant Step 1
        eps_z = jr.normal(key, shape=(N, D))
        z_t = z_t_m_1 @ F_t.T + eps_z @ chol_B_t.T
        return z_t

    #################################
    #        Initialisation         #
    #################################
    x0 = M_0_rvs(key_init, N)
    if conditional:
        x0 = tree_map(lambda x0_, xs0_: x0_.at[b_star[0]].set(xs0_), x0, tree_map(lambda x: x[0], x_star))

    # Compute initial weights and normalize
    log_w0 = Gamma_0(x0) - M_0_logpdf(x0)
    log_w0 = normalize(log_w0, log_space=True)

    #################################
    #        Forward pass           #
    #################################

    def body(carry, inp):
        log_w_t_m_1, x_t_m_1 = carry
        M_t_params, Gamma_params_t, x_star_t, b_star_t_m_1, b_star_t, key_t = inp

        key_z_t, key_resampling_t, key_eta_t = jr.split(key_t, 3)

        # Step 1: propose z_t candidates from the exact z transition
        z_t_hat = z_transition_rvs(key_z_t, x_t_m_1, M_t_params)

        # Step 2: compute auxiliary/predictive weights before drawing eta_t
        obs_t, _, _, dt = M_t_params
        _, eta_t_m_1 = x_t_m_1
        log_pred_t = predictive_obs_logpdf(z_t_hat, eta_t_m_1, obs_t, psi, chol_Q_eta, chol_R, dt)
        log_aux_w_t = normalize(log_w_t_m_1 + log_pred_t, log_space=True)
        w_aux_t = jnp.exp(log_aux_w_t)

        # Step 3: resample ancestors using the predictive weights
        A_t = resampling_func(key_resampling_t, w_aux_t, b_star_t_m_1, b_star_t, conditional)
        x_t_m_1 = tree_map(lambda x: jnp.take(x, A_t, axis=0), x_t_m_1)
        z_t = jnp.take(z_t_hat, A_t, axis=0)

        # Step 4--6: draw eta_t from the conditional guided proposal
        eta_t_m_1 = x_t_m_1[1]
        eta_t = mid_price_proposal(key_eta_t, z_t, eta_t_m_1, obs_t, psi, chol_Q_eta, chol_R, dt)
        x_t = (z_t, eta_t)

        if conditional:
            x_t = tree_map(lambda xt_, xs_t_: xt_.at[b_star_t].set(xs_t_), x_t, x_star_t)

        # Fully adapted after auxiliary resampling: equal filtering weights.
        log_w_t = -jnp.log(N) * jnp.ones((N,))
        # Return next step
        next_carry = log_w_t, x_t
        save = log_w_t, A_t, x_t

        return next_carry, save

    inputs = (M_t_params, Gamma_params, tree_map(lambda x: x[1:], x_star), b_star[:-1], b_star[1:], keys_forward)
    _, (log_ws, As, xs) = jax.lax.scan(body, (log_w0, x0), inputs)

    log_ws = jnp.insert(log_ws, 0, log_w0, axis=0)
    xs = tree_map(lambda xs_, x0_: jnp.insert(xs_, 0, x0_, axis=0), xs, x0)

    if backward:
        xs, Bs = backward_sampling_pass(key_backward, Gamma_t, Gamma_params, b_star[-1], xs, log_ws, ancestor_move_func)
    else:
        xs, Bs = backward_scanning_pass(key_backward, As, b_star[-1], xs, log_ws[-1], ancestor_move_func)

    return xs, Bs, log_ws


def _obs_var(chol_R: Array, bond_idx: Array):
    """
    Returns observation-noise variance for bond_idx.

    chol_R may be either:
        (D,)    vector of observation standard deviations
        (D, D)  Cholesky factor of observation covariance
    """
    if chol_R.ndim == 1:
        return chol_R[bond_idx] ** 2

    R = chol_R @ chol_R.T
    return R[bond_idx, bond_idx]


def _logdiffexp(a: Array, b: Array):
    """
    Computes log(exp(a) - exp(b)), assuming a >= b.
    """
    return jnp.where(b < a, a + jnp.log1p(-jnp.exp(b - a)), -jnp.inf)


def predictive_obs_logpdf(
        z_t: Array,
        eta_t_m_1: Array,
        obs_t: tuple[Array],
        psi: Array,
        chol_Q_eta: Array,
        chol_R: Array,
        dt: Array,
):
    """
    Guéant Step-2 predictive log-weight:

        log p(obs_t | z_t, eta_{t-1})

    after integrating out eta_t and the observation noise.
    """
    bond_idx, event_type, alpha_i, obs_value = obs_t
    bond_idx = bond_idx.astype(jnp.int32)
    event_type = event_type.astype(jnp.int32)

    # variance
    Q = chol_Q_eta @ chol_Q_eta.T
    var_i = Q[bond_idx, bond_idx]
    var_eps = _obs_var(chol_R, bond_idx)
    var_tilde = var_i * dt + var_eps
    std_tilde = jnp.sqrt(var_tilde)

    # extraction
    eta_prev_i = eta_t_m_1[:, bond_idx]
    half_spread = psi[bond_idx] * jnp.exp(z_t[:, bond_idx])

    # case based evaluation
    case_0 = lambda: norm.logpdf(obs_value + half_spread, loc=eta_prev_i, scale=std_tilde)
    case_1 = lambda: norm.logpdf(obs_value - half_spread, loc=eta_prev_i, scale=std_tilde)
    case_2 = lambda: norm.logcdf(eta_prev_i - (obs_value + half_spread), loc=0.0, scale=std_tilde)
    case_3 = lambda: norm.logcdf((obs_value - half_spread) - eta_prev_i, loc=0.0, scale=std_tilde)

    def case_4():
        log_hi = norm.logcdf((obs_value + alpha_i) - eta_prev_i, loc=0.0, scale=std_tilde)
        log_lo = norm.logcdf((obs_value - alpha_i) - eta_prev_i, loc=0.0, scale=std_tilde)
        return _logdiffexp(log_hi, log_lo)

    return jax.lax.switch(event_type, [case_0, case_1, case_2, case_3, case_4])


def mid_price_proposal(
        key: PRNGKey,
        z: Array,
        eta_prev: Array,
        obs: tuple[Array],
        psi: Array,
        chol_Q_eta: Array,
        chol_R: Array,
        dt: Array,
):
    """
    Observation-guided proposal for eta_t.

    Parameters
    ----------
    z:        (N, D)
    eta_prev: (N, D)
    obs:      Tuple (bond_idx, event_type, alpha_i, obs_value)

    Returns
    -------
    eta:     (N, D)
    """
    # house keeping
    N, D = z.shape
    key_i, key_not_i, key_tilde = jr.split(key, 3)

    bond_idx, event_type, alpha_i, obs_value = obs
    bond_idx = bond_idx.astype(jnp.int32)
    event_type = event_type.astype(jnp.int32)

    # variance
    Q = chol_Q_eta @ chol_Q_eta.T
    var_i = Q[bond_idx, bond_idx]
    var_eps = _obs_var(chol_R, bond_idx)
    var_tilde = var_i * dt + var_eps
    std_tilde = jnp.sqrt(var_tilde)

    # extraction
    eta_prev_i = eta_prev[:, bond_idx]
    half_spread = psi[bond_idx] * jnp.exp(z[:, bond_idx])

    standardise = lambda x: (x - eta_prev_i) / std_tilde

    # eta_i_tilde = eta_i + eps_i
    case_0 = lambda: obs_value + half_spread
    case_1 = lambda: obs_value - half_spread
    case_2 = lambda: eta_prev_i + std_tilde * jr.truncated_normal(
        key_tilde,
        lower=standardise(obs_value + half_spread),
        upper=jnp.inf,
        shape=(N,),
    )
    case_3 = lambda: eta_prev_i + std_tilde * jr.truncated_normal(
        key_tilde,
        lower=-jnp.inf,
        upper=standardise(obs_value - half_spread),
        shape=(N,),
    )
    case_4 = lambda: eta_prev_i + std_tilde * jr.truncated_normal(
        key_tilde,
        lower=standardise(obs_value - alpha_i),
        upper=standardise(obs_value + alpha_i),
        shape=(N,),
    )

    eta_i_tilde = jax.lax.switch(event_type, [case_0, case_1, case_2, case_3, case_4])

    # eta_i | eta_i + eps_i, eta_{i,t-1}
    mean_i = (var_i * dt * eta_i_tilde + var_eps * eta_prev_i) / var_tilde
    var_post_i = (var_i * dt * var_eps) / var_tilde
    eta_i = mean_i + jnp.sqrt(var_post_i) * jr.normal(key_i, shape=(N,))

    # eta_{-i} | eta_i under eta_t | eta_{t-1} ~ N(eta_{t-1}, dt Q)
    eps = jnp.sqrt(dt) * (jr.normal(key_not_i, shape=(N, D)) @ chol_Q_eta.T)
    eps_i = eps[:, bond_idx]
    delta_i = eta_i - eta_prev[:, bond_idx]
    beta = Q[:, bond_idx] / var_i
    eta = eta_prev + (delta_i[:, None] * beta[None, :]) + eps - (eps_i[:, None] * beta[None, :])
    eta = eta.at[:, bond_idx].set(eta_i)

    return eta


def Mt_tilde_logpdf(
        x_t_m_1,
        x_t,
        params,
        observation_logpdf: Callable,
        chol_Q_eta: Array,
        chol_R: Array,
        psi: Array,
):
    """
    Log-density of the actual guided proposal:

        q_tilde(z_t, eta_t | z_{t-1}, eta_{t-1}, obs_t)

    This includes the eta prior transition term. That term cancels in the
    forward importance weight, but it is part of the proposal density.
    """
    z_t_m_1, eta_t_m_1 = x_t_m_1
    z_t, eta_t = x_t
    D = z_t.shape[-1]

    obs_t, F_t, chol_B_t, dt = params

    # inverse cholesky factors
    inv_chol_B = solve_triangular(chol_B_t, jnp.eye(D), lower=True)
    inv_chol_eta_dt = solve_triangular(jnp.sqrt(dt) * chol_Q_eta, jnp.eye(D), lower=True)

    # prior log pdfs
    log_q_z = mvn_logpdf(z_t, z_t_m_1 @ F_t.T, None, chol_inv=inv_chol_B, constant=False)
    log_prior_eta = mvn_logpdf(eta_t, eta_t_m_1, None, chol_inv=inv_chol_eta_dt, constant=False)
    log_g = observation_logpdf(z_t, eta_t, obs_t, psi, chol_R)

    # guided correction
    log_pred = predictive_obs_logpdf(z_t, eta_t_m_1, obs_t, psi, chol_Q_eta, chol_R, dt)

    return log_q_z + log_prior_eta + log_g - log_pred
