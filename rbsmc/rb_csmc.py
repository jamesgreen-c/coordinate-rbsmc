"""Rao-Blackwellised conditional SMC for coordinate-observation models."""
from enum import Enum
from typing import Any, Callable, NamedTuple

import jax
import jax.random as jr
import jax.numpy as jnp

from chex import Array, PRNGKey
from jax.scipy.linalg import solve_triangular
from jax.tree_util import tree_leaves, tree_map, tree_structure, tree_unflatten

from rbsmc.bayesian.smc import Reference
from rbsmc.utils.common import barker_move
from rbsmc.utils.resamplings import normalize


class BackwardMode(str, Enum):
    """Available ways to select the retained coordinate path."""

    ANCESTRAL = "ancestral"
    FFBSI = "ffbsi"
    REDUCED = "reduced"


class GaussianState(NamedTuple):
    """Particle-dependent means and shared covariances for Gaussian blocks."""

    means: Any
    covariances: Any


class GaussianPrediction(NamedTuple):
    """Affine Gaussian prediction returned by the supplied ``M_t`` function.

    Each leaf represents ``X_t = F_t X_{t-1} + a_t + eps_t``.
    ``cross_covariances`` contains
    ``Cov(X_{t-1}, X_t | u_{0:t-1})``.
    """

    means: Any
    covariances: Any
    cross_covariances: Any
    transitions: Any
    offsets: Any


class ParticleSystem(NamedTuple):
    """Quantities recorded by the forward RB particle filter."""

    ancestors: Array
    coordinates: Any
    means: Any
    covariances: Any
    log_weights: Array


def _unpack_function(specification, name: str):
    """Return ``(function, time_inputs)`` from a kernel function specification."""
    if callable(specification):
        return specification, None
    if (
            isinstance(specification, tuple)
            and len(specification) == 2
            and callable(specification[0])):
        return specification
    raise TypeError(f"{name} must be callable or a (callable, time_inputs) tuple.")


def kernel(
        key: PRNGKey,
        ref: Reference,
        indices: Array,
        M_0: Callable,
        G_0: Callable,
        M_t: tuple[Callable, Any],
        G_t: tuple[Callable, Any] | Callable,
        resampling_func: Callable,
        ancestor_move_func: Callable,
        N: int,
        backward: bool = False,
        backward_mode: BackwardMode | str | None = None,
        conditional: bool = True,
    ):
    """Run one Rao-Blackwellised conditional SMC sweep.

    ``N`` is the number of free particles, so the particle system contains
    ``N + 1`` particles. ``M_0``, ``G_0``, ``M_t`` and ``G_t`` have the same
    role as their counterparts in the full-state kernels. ``M_t`` is supplied
    as ``(transition_function, transition_inputs)``; ``G_t`` may similarly be
    supplied as ``(potential_function, potential_inputs)``.

    ``backward`` is retained for compatibility: ``False`` selects ancestral
    tracing and ``True`` selects FFBSi. Set ``backward_mode="reduced"`` for the
    reduced-dimensional backward sampler.

    Returns
    -------
    ref:
        Full state trajectory and its selected particle slots. The reduced
        coordinates required by the next RB sweep are extracted from the full
        trajectory, so they are not stored a second time.
    log_weights:
        Normalised filtering log-weights.
    """
    mode = _resolve_backward_mode(backward, backward_mode)
    M_t_func, M_t_params = _unpack_function(M_t, "M_t")
    G_t_func, G_t_params = _unpack_function(G_t, "G_t")

    ref_coordinates = _extract_path_coordinates(ref.trajectory, indices)
    ref_means, _ = condition_on_coordinates(ref_coordinates, indices, M_0, M_t_func, M_t_params)

    key_backward, particles = forward_pass(key, ref_coordinates, ref_means, ref.ancestors, indices,
                                           M_0, G_0, M_t_func, M_t_params, G_t_func, G_t_params,
                                           resampling_func, N, conditional)
    b_star_T = ref.ancestors[-1]

    if mode == BackwardMode.ANCESTRAL:
        ref = ancestral_pass(key_backward, particles, indices, M_0, M_t_func, M_t_params,
                             ancestor_move_func, conditional, b_star_T)
    elif mode == BackwardMode.FFBSI:
        ref = ffbsi_pass(key_backward, particles, indices, M_t_func, M_t_params,
                         ancestor_move_func, conditional, b_star_T)
    else:
        ref = reduced_pass(key_backward, particles, indices, M_0, M_t_func, M_t_params,
                           ancestor_move_func, conditional, b_star_T)

    return ref, particles.log_weights


def forward_pass(
        key: PRNGKey,
        ref_coordinates: Any,
        ref_means: Any,
        b_star: Array,
        indices: Array,
        M_0: Callable,
        G_0: Callable,
        M_t: Callable,
        M_t_params: Any,
        G_t: Callable,
        G_t_params: Any,
        resampling_func: Callable,
        N: int,
        conditional: bool = True,
    ):
    """Run the forward RB-CSMC pass and retain coordinates explicitly."""
    T = indices.shape[0]
    num_particles = N + 1
    key_init, key_loop, key_backward = jr.split(key, 3)

    initial = M_0(num_particles)
    u_0 = _sample_coordinates(key_init, indices[0], initial.means, initial.covariances)

    if conditional:
        u_0 = tree_map(lambda u, u_star: u.at[b_star[0]].set(u_star[0]), u_0, ref_coordinates)
    state_0 = _condition_state(indices[0], u_0, initial.means, initial.covariances)

    if conditional:
        means_0 = tree_map(
            lambda means, means_star: means.at[b_star[0]].set(means_star[0]),
            state_0.means, ref_means
        )
        state_0 = GaussianState(means_0, state_0.covariances)

    log_w_0 = normalize(G_0(state_0.means), log_space=True)
    w_0 = jnp.exp(log_w_0)

    def body(carry, inp):
        w_t_m_1, state_t_m_1 = carry
        M_t_params_t, G_t_params_t, key_t, idx_t, u_star_t, means_star_t, b_star_t_m_1, b_star_t = inp
        key_proposal_t, key_resampling_t = jr.split(key_t)

        A_t = resampling_func(key_resampling_t, w_t_m_1, b_star_t_m_1, b_star_t, conditional)
        means_t_m_1 = tree_map(lambda x: jnp.take(x, A_t, axis=0), state_t_m_1.means)
        prediction = M_t(means_t_m_1, state_t_m_1.covariances, M_t_params_t)
        u_t = _sample_coordinates(key_proposal_t, idx_t, prediction.means, prediction.covariances)

        if conditional:
            u_t = tree_map(lambda u, u_star: u.at[b_star_t].set(u_star), u_t, u_star_t)
        state_t = _condition_state(idx_t, u_t, prediction.means, prediction.covariances)

        if conditional:
            means_t = tree_map(
                lambda means, means_star: means.at[b_star_t].set(means_star),
                state_t.means, means_star_t
            )
            state_t = GaussianState(means_t, state_t.covariances)

        log_w_t = normalize(G_t(means_t_m_1, state_t.means, G_t_params_t), log_space=True)
        w_t = jnp.exp(log_w_t)
        return (w_t, state_t), (A_t, u_t, state_t.means, state_t.covariances, log_w_t)

    keys_loop = jr.split(key_loop, T - 1)
    scan_inps = (
        M_t_params,
        G_t_params,
        keys_loop,
        indices[1:],
        tree_map(lambda x: x[1:], ref_coordinates),
        tree_map(lambda x: x[1:], ref_means),
        b_star[:-1],
        b_star[1:],
    )
    _, (ancestors, coordinates, means, covariances, log_weights) = jax.lax.scan(body, 
                                                                                (w_0, state_0), 
                                                                                scan_inps)

    _prepend = lambda x, x_0: jnp.concatenate((x_0[None], x), axis=0)
    coordinates = tree_map(_prepend, coordinates, u_0)
    means = tree_map(_prepend, means, state_0.means)
    covariances = tree_map(_prepend, covariances, state_0.covariances)
    log_weights = _prepend(log_weights, log_w_0)
    # log_weights = jnp.concatenate((log_w_0[None], log_weights), axis=0)

    particles = ParticleSystem(ancestors, coordinates, means, covariances, log_weights)
    return key_backward, particles


def ancestral_pass(
        key: PRNGKey,
        particles: ParticleSystem,
        indices: Array,
        M_0: Callable,
        M_t: Callable,
        M_t_params: Any,
        ancestor_move_func: Callable,
        conditional: bool = True,
        b_star_T: int = 0,
    ):
    """Trace one forward genealogy, then draw the full Gaussian trajectory."""
    key_terminal, key_gaussian = jr.split(key)
    B_T = _sample_terminal(key_terminal, particles.log_weights[-1], 
                           ancestor_move_func, conditional, b_star_T)

    def body(B_t, A_t):
        B_t_m_1 = A_t[B_t]
        return B_t_m_1, B_t_m_1

    _, Bs_reverse = jax.lax.scan(body, B_T, particles.ancestors[::-1])
    Bs = jnp.concatenate((Bs_reverse[::-1], jnp.asarray(B_T)[None]), axis=0)
    coordinates = _gather_path(particles.coordinates, Bs)
    means, covariances = condition_on_coordinates(coordinates, indices, M_0, M_t, M_t_params)
    trajectory = conditional_gaussian_pass(
        key_gaussian, 
        coordinates, means, covariances, indices, 
        M_t, M_t_params
    )
    return Reference(trajectory, Bs)


def ffbsi_pass(
        key: PRNGKey,
        particles: ParticleSystem,
        indices: Array,
        M_t: Callable,
        M_t_params: Any,
        ancestor_move_func: Callable,
        conditional: bool = True,
        b_star_T: int = 0,
    ):
    """Apply full-state Forward Filtering Backward Simulation."""
    T = particles.log_weights.shape[0]
    key_terminal, key_terminal_simulation, key_loop = jr.split(key, 3)
    B_T = _sample_terminal(key_terminal, particles.log_weights[-1], 
                           ancestor_move_func, conditional, b_star_T)

    mean_T = tree_map(lambda x: x[-1, B_T], particles.means)
    covariance_T = tree_map(lambda x: x[-1], particles.covariances)
    x_T = _simulate_tree(key_terminal_simulation, mean_T, covariance_T)
    u_T = tree_map(lambda x: x[-1, B_T], particles.coordinates)

    def body(x_t, inp):
        key_t, means_t_m_1, covariances_t_m_1, log_w_t_m_1, M_t_params_t, u_particles = inp
        key_index, key_simulation = jr.split(key_t)

        prediction = M_t(means_t_m_1, covariances_t_m_1, M_t_params_t)
        log_M_t = _prediction_logpdf(x_t, prediction.means, prediction.covariances)

        weights = normalize(log_w_t_m_1 + log_M_t)
        B_t_m_1 = jr.choice(key_index, weights.shape[0], p=weights, shape=())

        means_smooth, covariances_smooth = _smoothing_statistics(means_t_m_1, 
                                                                 covariances_t_m_1, 
                                                                 x_t, 
                                                                 prediction)
        mean_smooth = tree_map(lambda x: x[B_t_m_1], means_smooth)

        x_t_m_1 = _simulate_tree(key_simulation, mean_smooth, covariances_smooth)
        u_t_m_1 = tree_map(lambda x: x[B_t_m_1], u_particles)
        return x_t_m_1, (x_t_m_1, u_t_m_1, B_t_m_1)

    keys_loop = jr.split(key_loop, T - 1)
    scan_inps = (
        keys_loop,
        tree_map(lambda x: x[-2::-1], particles.means),
        tree_map(lambda x: x[-2::-1], particles.covariances),
        particles.log_weights[-2::-1],
        tree_map(lambda x: x[::-1], M_t_params),
        tree_map(lambda x: x[-2::-1], particles.coordinates),
    )
    _, (trajectory_reverse, coordinates_reverse, Bs_reverse) = jax.lax.scan(body, x_T, scan_inps)

    trajectory = tree_map(
        lambda x, x_T: jnp.concatenate((x[::-1], x_T[None]), axis=0),
        trajectory_reverse, x_T
    )
    coordinates = tree_map(
        lambda x, u_T: jnp.concatenate((x[::-1], jnp.asarray(u_T)[None]), axis=0),
        coordinates_reverse, u_T
    )
    Bs = jnp.concatenate((Bs_reverse[::-1], jnp.asarray(B_T)[None]), axis=0)
    trajectory = _set_path_coordinates(trajectory, coordinates, indices)
    return Reference(trajectory, Bs)


def reduced_pass(
        key: PRNGKey,
        particles: ParticleSystem,
        indices: Array,
        M_0: Callable,
        M_t: Callable,
        M_t_params: Any,
        ancestor_move_func: Callable,
        conditional: bool = True,
        b_star_T: int = 0,
    ):
    """Select only the coordinate path using the quadratic future message."""
    T = particles.log_weights.shape[0]
    key_terminal, key_loop, key_gaussian = jr.split(key, 3)
    B_T = _sample_terminal(
        key_terminal, particles.log_weights[-1], ancestor_move_func,
        conditional, b_star_T
    )
    u_T = tree_map(lambda x: x[-1, B_T], particles.coordinates)
    omega_T = tree_map(lambda x: jnp.zeros_like(x[-1]), particles.covariances)
    eta_T = tree_map(lambda x: jnp.zeros_like(x[-1, 0]), particles.means)

    def body(carry, inp):
        omega_t, eta_t, u_t = carry
        key_t, means_t_m_1, covariances_t_m_1, log_w_t_m_1, M_t_params_t, idx_t, u_particles = inp

        prediction = M_t(means_t_m_1, covariances_t_m_1, M_t_params_t)
        omega_t_m_1, eta_t_m_1 = _update_information(omega_t, eta_t, u_t, prediction, idx_t)
        log_message = _information_logpdf(means_t_m_1, omega_t_m_1, eta_t_m_1)

        weights = normalize(log_w_t_m_1 + log_message)
        B_t_m_1 = jr.choice(key_t, weights.shape[0], p=weights, shape=())

        u_t_m_1 = tree_map(lambda x: x[B_t_m_1], u_particles)
        return (omega_t_m_1, eta_t_m_1, u_t_m_1), (u_t_m_1, B_t_m_1)

    keys_loop = jr.split(key_loop, T - 1)
    scan_inps = (
        keys_loop,
        tree_map(lambda x: x[-2::-1], particles.means),
        tree_map(lambda x: x[-2::-1], particles.covariances),
        particles.log_weights[-2::-1],
        tree_map(lambda x: x[::-1], M_t_params),
        indices[:0:-1],
        tree_map(lambda x: x[-2::-1], particles.coordinates),
    )
    _, (coordinates_reverse, Bs_reverse) = jax.lax.scan(body, (omega_T, eta_T, u_T), scan_inps)

    coordinates = tree_map(
        lambda x, u_T: jnp.concatenate((x[::-1], jnp.asarray(u_T)[None]), axis=0),
        coordinates_reverse, u_T
    )
    Bs = jnp.concatenate((Bs_reverse[::-1], jnp.asarray(B_T)[None]), axis=0)
    means, covariances = condition_on_coordinates(coordinates, indices, M_0, M_t, M_t_params)
    trajectory = conditional_gaussian_pass(key_gaussian, coordinates, 
                                           means, covariances, indices, 
                                           M_t, M_t_params)
    return Reference(trajectory, Bs)


def condition_on_coordinates(
        coordinates: Any,
        indices: Array,
        M_0: Callable,
        M_t: Callable,
        M_t_params: Any,
    ):
    """Run the RB recursion along a coordinate path to recover Gaussian states."""
    initial = M_0(1)
    mean_pred_0 = tree_map(lambda x: x[0], initial.means)
    mean_0 = tree_map(
        lambda u, mean, covariance: _condition_mean(indices[0], u[0], mean, covariance),
        coordinates, mean_pred_0, initial.covariances
    )
    covariance_0 = tree_map(
        lambda covariance: _condition_covariance(indices[0], covariance),
        initial.covariances
    )

    def body(carry, inp):
        mean_t_m_1, covariance_t_m_1 = carry
        M_t_params_t, idx_t, u_t = inp

        batched_mean = tree_map(lambda x: x[None], mean_t_m_1)
        prediction = M_t(batched_mean, covariance_t_m_1, M_t_params_t)
        
        mean_pred_t = tree_map(lambda x: x[0], prediction.means)
        mean_t = tree_map(
            lambda u, mean, covariance: _condition_mean(idx_t, u, mean, covariance),
            u_t, mean_pred_t, prediction.covariances
        )
        covariance_t = tree_map(
            lambda covariance: _condition_covariance(idx_t, covariance),
            prediction.covariances
        )
        return (mean_t, covariance_t), (mean_t, covariance_t)

    scan_inps = (M_t_params, indices[1:], tree_map(lambda x: x[1:], coordinates))
    _, (means, covariances) = jax.lax.scan(body, (mean_0, covariance_0), scan_inps)
    means = tree_map(lambda x, x_0: jnp.concatenate((x_0[None], x), axis=0), means, mean_0)
    covariances = tree_map(lambda x, x_0: jnp.concatenate((x_0[None], x), axis=0), covariances, covariance_0)
    return means, covariances


def conditional_gaussian_pass(
        key: PRNGKey,
        coordinates: Any,
        means: Any,
        covariances: Any,
        indices: Array,
        M_t: Callable,
        M_t_params: Any,
    ):
    """Draw the full trajectory conditional on a selected coordinate path."""
    T = indices.shape[0]
    key_terminal, key_loop = jr.split(key)
    mean_T = tree_map(lambda x: x[-1], means)
    covariance_T = tree_map(lambda x: x[-1], covariances)
    x_T = _simulate_tree(key_terminal, mean_T, covariance_T)

    def body(x_t, inp):
        key_t, mean_t_m_1, covariance_t_m_1, M_t_params_t = inp

        batched_mean = tree_map(lambda x: x[None], mean_t_m_1)
        prediction = M_t(batched_mean, covariance_t_m_1, M_t_params_t)

        means_smooth, covariances_smooth = _smoothing_statistics(batched_mean, 
                                                                 covariance_t_m_1, 
                                                                 x_t, 
                                                                 prediction)
        mean_smooth = tree_map(lambda x: x[0], means_smooth)
        
        x_t_m_1 = _simulate_tree(key_t, mean_smooth, covariances_smooth)
        return x_t_m_1, x_t_m_1

    keys_loop = jr.split(key_loop, T - 1)
    scan_inps = (
        keys_loop,
        tree_map(lambda x: x[-2::-1], means),
        tree_map(lambda x: x[-2::-1], covariances),
        tree_map(lambda x: x[::-1], M_t_params),
    )
    _, trajectory_reverse = jax.lax.scan(body, x_T, scan_inps)
    trajectory = tree_map(
        lambda x, x_T: jnp.concatenate((x[::-1], x_T[None]), axis=0),
        trajectory_reverse, x_T
    )
    return _set_path_coordinates(trajectory, coordinates, indices)


def _condition_state(i: Array, coordinates: Any, means: Any, covariances: Any):
    conditioned_means = tree_map(
        lambda u, mean, covariance: _condition_means(i, u, mean, covariance),
        coordinates, means, covariances
    )
    conditioned_covariances = tree_map(
        lambda covariance: _condition_covariance(i, covariance), covariances
    )
    return GaussianState(conditioned_means, conditioned_covariances)


def _condition_means(i: Array, u: Array, mean_pred: Array, covariance_pred: Array):
    gain = covariance_pred[:, i] / covariance_pred[i, i]
    return mean_pred + (u - mean_pred[:, i])[:, None] * gain[None, :]


def _condition_mean(i: Array, u: Array, mean_pred: Array, covariance_pred: Array):
    gain = covariance_pred[:, i] / covariance_pred[i, i]
    return mean_pred + (u - mean_pred[i]) * gain


def _condition_covariance(i: Array, covariance_pred: Array):
    column = covariance_pred[:, i]
    covariance = covariance_pred - jnp.outer(column, column) / covariance_pred[i, i]
    covariance = covariance.at[i, :].set(0.0)
    covariance = covariance.at[:, i].set(0.0)
    return 0.5 * (covariance + covariance.T)


def _sample_coordinates(key: PRNGKey, i: Array, means: Any, covariances: Any):
    keys = _split_key_like(key, means)
    return tree_map(
        lambda key, m, cov: m[:, i] + jr.normal(key, shape=(m.shape[0],)) * jnp.sqrt(cov[i, i]),
        keys, means, covariances
    )


def _smoothing_statistics(
        means: Any,
        covariances: Any,
        x_t: Any,
        prediction: GaussianPrediction,
    ):
    treedef = tree_structure(means)
    means_smooth = []
    covariances_smooth = []

    for mean, covariance, x, mean_pred, covariance_pred, cross in zip(
            tree_leaves(means), tree_leaves(covariances), tree_leaves(x_t),
            tree_leaves(prediction.means), tree_leaves(prediction.covariances),
            tree_leaves(prediction.cross_covariances)):
        gain = jnp.linalg.solve(covariance_pred.T, cross.T).T
        means_smooth.append(mean + (x - mean_pred) @ gain.T)
        covariance_smooth = covariance - gain @ cross.T
        covariances_smooth.append(0.5 * (covariance_smooth + covariance_smooth.T))

    return tree_unflatten(treedef, means_smooth), tree_unflatten(treedef, covariances_smooth)


def _prediction_logpdf(x_t: Any, means: Any, covariances: Any):
    value = 0.0
    log_2pi = jnp.log(2.0 * jnp.pi)

    for x, mean, covariance in zip(tree_leaves(x_t), tree_leaves(means), tree_leaves(covariances)):
        chol = jnp.linalg.cholesky(covariance)
        residual = x - mean
        whitened = solve_triangular(chol, residual.T, lower=True).T
        value += -0.5 * (
            jnp.sum(whitened ** 2, axis=-1)
            + 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
            + x.shape[-1] * log_2pi
        )

    return value


def _update_information(
        omega_t: Any,
        eta_t: Any,
        u_t: Any,
        prediction: GaussianPrediction,
        i_t: Array,
    ):
    treedef = tree_structure(omega_t)
    omega_t_m_1 = []
    eta_t_m_1 = []

    for omega, eta, u, covariance_pred, transition, offset in zip(
            tree_leaves(omega_t), tree_leaves(eta_t), tree_leaves(u_t),
            tree_leaves(prediction.covariances), tree_leaves(prediction.transitions),
            tree_leaves(prediction.offsets)
        ):
        observation = transition[i_t, :]
        observation_offset = offset[i_t]

        variance = covariance_pred[i_t, i_t]
        gain = covariance_pred[:, i_t] / variance
        
        gamma = transition - jnp.outer(gain, observation)
        conditioned_offset = offset - gain * observation_offset
        displacement = conditioned_offset + gain * u

        next_omega = jnp.outer(observation, observation) / variance + gamma.T @ omega @ gamma
        next_eta = observation * (u - observation_offset) / variance + gamma.T @ (eta - omega @ displacement)

        omega_t_m_1.append(0.5 * (next_omega + next_omega.T))
        eta_t_m_1.append(next_eta)

    return tree_unflatten(treedef, omega_t_m_1), tree_unflatten(treedef, eta_t_m_1)


def _information_logpdf(means: Any, omegas: Any, etas: Any):
    value = 0.0
    for mean, omega, eta in zip(
            tree_leaves(means), tree_leaves(omegas), tree_leaves(etas)):
        value += mean @ eta - 0.5 * jnp.einsum("ni,ij,nj->n", mean, omega, mean)
    return value


def _sample_terminal(
        key: PRNGKey,
        log_weights: Array,
        ancestor_move_func: Callable,
        conditional: bool,
        b_star_T: int,
    ):
    weights = normalize(log_weights)
    if conditional:
        index, _ = ancestor_move_func(key, weights, b_star_T)
    else:
        index, _ = barker_move(key, weights, None)
    return index


def _simulate_tree(key: PRNGKey, means: Any, covariances: Any):
    keys = _split_key_like(key, means)
    return tree_map(_simulate, keys, means, covariances)


def _simulate(key: PRNGKey, mean: Array, covariance: Array):
    """Sample from a potentially singular Gaussian distribution."""
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
    eigenvalues = jnp.maximum(eigenvalues, 0.0)
    eps = jr.normal(key, mean.shape)
    factor = eigenvectors * jnp.sqrt(eigenvalues)[None, :]
    return mean + eps @ factor.T


def _gather_path(values: Any, indices: Array):
    times = jnp.arange(indices.shape[0])
    return tree_map(lambda x: x[times, indices], values)


def _extract_path_coordinates(trajectory: Any, indices: Array):
    """Extract each scheduled coordinate from a full state trajectory."""
    return _gather_path(trajectory, indices)


def _set_path_coordinates(trajectory: Any, coordinates: Any, indices: Array):
    times = jnp.arange(indices.shape[0])
    return tree_map(lambda x, u: x.at[times, indices].set(u), trajectory, coordinates)


def _split_key_like(key: PRNGKey, tree: Any):
    leaves = tree_leaves(tree)
    keys = jr.split(key, len(leaves))
    return tree_unflatten(tree_structure(tree), keys)


def _resolve_backward_mode(
        backward: bool,
        backward_mode: BackwardMode | str | None,
    ):
    if backward_mode is None:
        return BackwardMode.FFBSI if backward else BackwardMode.ANCESTRAL

    mode = BackwardMode(backward_mode)
    if backward and mode != BackwardMode.FFBSI:
        raise ValueError("backward=True is only compatible with backward_mode='ffbsi'.")
    return mode
