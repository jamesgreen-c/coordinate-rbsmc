"""
Implements the Rao-Blackwellised cSMC kernel that accepts tuple reference trajectories.
"""
from typing import Any, Callable

import jax
from chex import Array, PRNGKey
from jax import numpy as jnp
from jax.tree_util import tree_leaves, tree_map, tree_structure, tree_unflatten

from rbsmc.utils.common import barker_move
from rbsmc.utils.resamplings import normalize


def kernel(
        key: PRNGKey,
        x_star: Array,
        b_star: Array,
        indices: Array,
        M_0_rvs: Callable,
        G_0: Callable,
        M_t_rvs: Callable,
        G_t: Callable,
        M_t_logpdf: Callable,
        rts_func: Callable,
        inps: Any,
        resampling_func: Callable,
        ancestor_move_func: Callable,
        N: int,
        backward: bool = False,
        conditional: bool = True
    ):
    """
    Rao-Blackwellised cSMC kernel.

    Parameters
    ----------
    key:                 Random number generator key.
    x_star:              Reference trajectory.
    b_star:              Particle indices occupied by the reference trajectory.
    indices:             Coordinate sampled at each time.
    M_0_rvs:             Initial proposal sampler.
    G_0:                 Initial potential.
    M_t_rvs:             Transition proposal sampler.
    G_t:                 Incremental potential.
    M_t_logpdf:          Predictive transition log-density used in backward sampling.
    rts_func:            Function returning conditional smoothing means and covariances.
    inps:                Transition inputs for times 1, ..., T - 1.
    resampling_func:     Resampling scheme.
    ancestor_move_func:  Function used to move the terminal reference ancestor.
    N:                   Number of particles, including the reference particle.
    backward:            Whether to use backward sampling instead of ancestral tracing.
    conditional:         Whether to run conditional SMC.

    Returns
    -------
    trajectory:  Sampled full-state trajectory.
    Bs:          Selected particle indices.
    log_ws:      Normalised filtering log-weights.
    """
    As, key_backward, log_ws, xs, Ps = forward_pass(key, x_star, b_star, indices,
                                                    M_0_rvs, G_0, M_t_rvs, G_t,
                                                    inps, resampling_func, N, conditional)

    if backward:
        trajectory, Bs = backward_sampling_pass(key_backward, M_t_logpdf, rts_func, inps, 
                                                b_star[-1], xs, Ps, log_ws, 
                                                ancestor_move_func, conditional)
    else:
        trajectory, Bs = backward_scanning_pass(key_backward, As, rts_func, inps, 
                                                b_star[-1], xs, Ps, log_ws[-1], 
                                                ancestor_move_func, conditional)

    return trajectory, Bs, log_ws


def forward_pass(
        key: PRNGKey,
        x_star: Array,
        b_star: Array,
        indices: Array,
        M_0_rvs: Callable,
        G_0: Callable,
        M_t_rvs: Callable,
        G_t: Callable,
        inps: Any,
        resampling_func: Callable,
        N: int,
        conditional: bool = True
    ):
    """
    Run the forward Rao-Blackwellised cSMC pass.

    Each particle contains a sampled coordinate and the conditional Gaussian
    distribution of the remaining coordinates.
    """
    T = tree_leaves(x_star)[0].shape[0]
    key_init, key_loop, key_backward = jax.random.split(key, 3)

    #################################
    #        Initialisation         #
    #################################
    u_0, mu_pred_0, P_pred_0 = M_0_rvs(key_init, N)

    if conditional:
        x_star_0 = tree_map(lambda x: x[0], x_star)
        u_0 = tree_map(lambda u, x: u.at[b_star[0]].set(x[..., indices[0]]), u_0, x_star_0)

    x_0 = tree_map(lambda u, mu, P: _marginalise_means(indices[0], u, mu, P), u_0, mu_pred_0, P_pred_0)
    P_0 = tree_map(lambda P: _marginalise_covs(indices[0], P), P_pred_0)

    # calculate weights
    log_w_0 = normalize(G_0(x_0), log_space=True)
    w_0 = jnp.exp(log_w_0)

    #################################
    #        Forward pass           #
    #################################
    def body(carry, inp):
        w_t_m_1, x_t_m_1, P_t_m_1 = carry
        inp_t, b_star_t_m_1, b_star_t, key_t, x_star_t, idx_t = inp

        key_proposal_t, key_resampling_t = jax.random.split(key_t)

        # resampling
        A_t = resampling_func(key_resampling_t, w_t_m_1, b_star_t_m_1, b_star_t, conditional)
        x_t_m_1 = tree_map(lambda x: jnp.take(x, A_t, axis=0), x_t_m_1)

        # propose particles for observation index
        u_t, mu_pred_t, P_pred_t = M_t_rvs(key_proposal_t,x_t_m_1, P_t_m_1, inp_t)

        # fix observed reference coordinate before weighting
        if conditional:
            u_t = tree_map(lambda u, x: u.at[b_star_t].set(x[..., idx_t]), u_t, x_star_t)

        # exact marginalisation
        x_t = tree_map(lambda u, mu, P: _marginalise_means(idx_t, u, mu, P), u_t, mu_pred_t, P_pred_t)
        P_t = tree_map(lambda P: _marginalise_covs(idx_t, P), P_pred_t)

        # weight calculation
        log_w_t = normalize(G_t(x_t_m_1, x_t, inp_t), log_space=True)
        w_t = jnp.exp(log_w_t)

        return (w_t, x_t, P_t), (log_w_t, A_t, x_t, P_t)

    # run forward cSMC
    keys_loop = jax.random.split(key_loop, T - 1)
    scan_inps = (inps, b_star[:-1], b_star[1:], keys_loop, tree_map(lambda x: x[1:], x_star), indices[1:])
    _, (log_ws, As, xs, Ps) = jax.lax.scan(body, (w_0, x_0, P_0), scan_inps)

    # insert initial weights and particles
    log_ws = jnp.insert(log_ws, 0, log_w_0, axis=0)
    xs = tree_map(lambda x, x0: jnp.insert(x, 0, x0, axis=0), xs, x_0)
    Ps = tree_map(lambda P, P0: jnp.insert(P, 0, P0, axis=0), Ps, P_0)

    return As, key_backward, log_ws, xs, Ps


def backward_sampling_pass(
        key: PRNGKey,
        M_t_logpdf: Callable,
        rts_func: Callable,
        inps: Any,
        b_star_T: Array,
        xs: Any,
        Ps: Any,
        log_ws: Array,
        ancestor_move_func: Callable,
        conditional: bool = True
    ):
    """
    Sample particle indices using Rao-Blackwellised backward weights and then
    sample the marginalised coordinates from their smoothing distributions.
    """
    T = tree_leaves(xs)[0].shape[0]
    move_key, simulation_key = jax.random.split(key)
    move_keys = jax.random.split(move_key, T)
    simulation_keys = jax.random.split(simulation_key, T)

    if conditional:
        B_T, _ = ancestor_move_func(move_keys[-1], normalize(log_ws[-1]), b_star_T)
    else:
        B_T, _ = barker_move(move_keys[-1], normalize(log_ws[-1]), None)

    x_T = tree_map(lambda x: x[-1, B_T], xs)
    P_T = tree_map(lambda P: P[-1], Ps)
    u_T = tree_map(_simulate, _split_key_like(simulation_keys[-1], x_T), x_T, P_T)

    def body(u_t, inp):
        op_key, sim_key, xs_t_m_1, P_t_m_1, log_w_t_m_1, inp_t = inp

        # backward weight calculation
        log_M_t = M_t_logpdf(xs_t_m_1, P_t_m_1, u_t, inp_t)
        w_t_m_1 = normalize(log_w_t_m_1 + log_M_t)

        # ancestor sampling
        B_t_m_1 = jax.random.choice(op_key, w_t_m_1.shape[0], p=w_t_m_1, shape=())

        # RTS smoothing
        ms_smooth, P_smooth = rts_func(xs_t_m_1, P_t_m_1, u_t, inp_t)
        m_smooth = tree_map(lambda m: m[B_t_m_1], ms_smooth)
        u_t_m_1 = tree_map(_simulate, _split_key_like(sim_key, m_smooth), m_smooth, P_smooth)

        return u_t_m_1, (u_t_m_1, B_t_m_1)

    scan_inps = (
        move_keys[:-1],
        simulation_keys[:-1],
        tree_map(lambda x: x[-2::-1], xs),
        tree_map(lambda P: P[-2::-1], Ps),
        log_ws[-2::-1],
        tree_map(lambda x: x[::-1], inps)
    )

    _, (trajectory, Bs) = jax.lax.scan(body, u_T, scan_inps)

    # insert terminal time sample and restore chronological ordering
    trajectory = tree_map(lambda x, xT: jnp.insert(x, 0, xT, axis=0)[::-1], trajectory, u_T)
    Bs = jnp.insert(Bs, 0, B_T, axis=0)[::-1]

    return trajectory, Bs


def backward_scanning_pass(
        key: PRNGKey,
        As: Array,
        rts_func: Callable,
        inps: Any,
        b_star_T: Array,
        xs: Any,
        Ps: Any,
        log_w_T: Array,
        ancestor_move_func: Callable,
        conditional: bool = True
    ):
    """
    Trace the ancestral particle path and sample the marginalised coordinates
    along that path using the conditional smoothing distributions.

    This differs from backward sampling only in how the particle indices are
    selected. Here they are fixed by the forward ancestry.
    """
    T = tree_leaves(xs)[0].shape[0]
    terminal_key, simulation_key = jax.random.split(key)
    simulation_keys = jax.random.split(simulation_key, T)

    if conditional:
        B_T, _ = ancestor_move_func(terminal_key, normalize(log_w_T), b_star_T)
    else:
        B_T, _ = barker_move(terminal_key, normalize(log_w_T), None)

    x_T = tree_map(lambda x: x[-1, B_T], xs)
    P_T = tree_map(lambda P: P[-1], Ps)
    u_T = tree_map(_simulate, _split_key_like(simulation_keys[-1], x_T), x_T, P_T)

    def body(carry, inp):
        B_t, u_t = carry
        sim_key, A_t, xs_t_m_1, P_t_m_1, inp_t = inp

        # ancestor selection
        B_t_m_1 = A_t[B_t]

        # RTS smoothing
        ms_smooth, P_smooth = rts_func(xs_t_m_1, P_t_m_1, u_t, inp_t)
        m_smooth = tree_map(lambda m: m[B_t_m_1], ms_smooth)
        u_t_m_1 = tree_map(_simulate, _split_key_like(sim_key, m_smooth), m_smooth, P_smooth)

        return (B_t_m_1, u_t_m_1), (u_t_m_1, B_t_m_1)

    scan_inps = (
        simulation_keys[:-1],
        As[::-1],
        tree_map(lambda x: x[-2::-1], xs),
        tree_map(lambda P: P[-2::-1], Ps),
        tree_map(lambda x: x[::-1], inps)
    )

    _, (trajectory, Bs) = jax.lax.scan(body, (B_T, u_T), scan_inps)

    # insert terminal sample and restore chronological ordering
    trajectory = tree_map(lambda x, xT: jnp.insert(x, 0, xT, axis=0)[::-1], trajectory,u_T)
    Bs = jnp.insert(Bs, 0, B_T, axis=0)[::-1]

    return trajectory, Bs


def _marginalise_means(
        i: Array,
        u: Array,
        mu_pred: Array,
        P_pred: Array
    ):
    """
    Condition a predictive Gaussian on the sampled coordinate u = x[i].

    Parameters
    ----------
    i:        Sampled coordinate index.
    u:        Sampled coordinate values with shape (N,).
    mu_pred:  Predictive means with shape (N, D).
    P_pred:   Predictive covariance with shape (D, D).

    Returns
    -------
    mu:  Conditional means with shape (N, D).
    """
    u = jnp.ravel(u)
    variance = jnp.maximum(P_pred[i, i], 1e-8)
    gain = P_pred[:, i] / variance
    return mu_pred + (u - mu_pred[:, i])[:, None] * gain[None, :]


def _marginalise_covs(i: Array, P_pred: Array):
    """
    Condition a predictive covariance on the sampled coordinate x[i].

    Parameters
    ----------
    i:       Sampled coordinate index.
    P_pred:  Predictive covariance with shape (D, D).

    Returns
    -------
    P:  Conditional covariance with shape (D, D).
    """
    variance = jnp.maximum(P_pred[i, i], 1e-8)
    column = P_pred[:, i]

    P = P_pred - jnp.outer(column, column) / variance
    P = P.at[i, :].set(0.0)
    P = P.at[:, i].set(0.0)

    return 0.5 * (P + P.T)


def _simulate(key: PRNGKey, mean: Array, covariance: Array):
    """Sample from a potentially singular Gaussian distribution."""
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
    eigenvalues = jnp.maximum(eigenvalues, 0.0)

    eps = jax.random.normal(key, mean.shape)
    factor = eigenvectors * jnp.sqrt(eigenvalues)[None, :]
    return mean + eps @ factor.T


def _split_key_like(key: PRNGKey, tree: Any):
    """Construct a PRNG-key pytree with the same structure as `tree`."""
    leaves = tree_leaves(tree)
    treedef = tree_structure(tree)
    keys = jax.random.split(key, len(leaves))
    return tree_unflatten(treedef, keys)