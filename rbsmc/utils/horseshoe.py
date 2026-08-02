import jax.numpy as jnp
import jax.random as jr

from jax import Array, vmap
from jax.random import PRNGKey
from jax.scipy.linalg import solve_triangular

from rbsmc.utils.inverse_gamma import inverse_gamma


class Horseshoe:

    @staticmethod
    def _other_indices(d: int, D: int) -> Array:
        return jnp.concatenate((jnp.arange(d), jnp.arange(d + 1, D)))

    @staticmethod
    def _sample_normal(key, precision: Array, linear_term: Array) -> Array:
        """
        Sample alpha ~ N(-C @ linear_term, C), where C = precision^{-1}.
        """
        D = precision.shape[0]
        chol_precision = jnp.linalg.cholesky(precision)
        tmp = solve_triangular(chol_precision, linear_term, lower=True)
        mean = -solve_triangular(chol_precision.T, tmp, lower=False)
        chol_C = solve_triangular(chol_precision.T, jnp.eye(D), lower=False)
        return mean + chol_C @ jr.normal(key, shape=(D,))

    @classmethod
    def init(cls, D: int):
        """
        Initialise the graphical-horseshoe Gibbs sampler.

        This is a valid starting state, not a joint draw from the prior, since the
        prior on the diagonal precision elements is improper.
        """

        Q = jnp.eye(D)
        beta = jnp.eye(D)
        llambda = jnp.ones((D, D))
        nu = jnp.ones((D, D))
        tau = jnp.array(1.0)
        xi = jnp.array(1.0)

        return Q, beta, llambda, nu, tau, xi

    @classmethod
    def _sample_column(
            cls,
            key,
            d: int,
            N: int,
            scatter: Array,
            beta: Array,
            Q: Array,
            llambda: Array,
            nu: Array,
            tau: Array,
        ):
        """
        Update row and column d of beta, together with their local shrinkage parameters.

        Here llambda[i, j] stores lambda_ij^2.
        """
        D = beta.shape[-1]
        idx = cls._other_indices(d, D)
        key_gamma, key_alpha, key_lambda, key_nu = jr.split(key, 4)

        # beta_{-d,-d}^{-1}, obtained from the current covariance Q = beta^{-1}
        Q_dd_inv = Q[jnp.ix_(idx, idx)] - jnp.outer(Q[idx, d], Q[idx, d]) / Q[d, d]

        # gamma ~ Gamma(N / 2 + 1, rate=scatter[d, d] / 2)
        gamma = jr.gamma(key_gamma, (N/2) + 1) * (2 / scatter[d, d])

        # alpha ~ N(-C scatter[-d,d], C)
        prior_precision = jnp.diag(1 / (llambda[idx, d] * tau**2))
        precision = scatter[d, d] * Q_dd_inv + prior_precision
        alpha = cls._sample_normal(key_alpha, precision, scatter[idx, d])

        # transform (alpha, gamma) back to the precision-matrix elements
        beta_dd = gamma + alpha @ Q_dd_inv @ alpha
        beta = beta.at[idx, d].set(alpha)
        beta = beta.at[d, idx].set(alpha)
        beta = beta.at[d, d].set(beta_dd)

        # lambda_ij^2 | ... and nu_ij | ...
        llambda_d = inverse_gamma(key_lambda, 1, (1/nu[idx, d]) + alpha**2 / (2 * tau**2))
        nu_d = inverse_gamma(key_nu, 1, 1 + (1/llambda_d))

        llambda = llambda.at[idx, d].set(llambda_d)
        llambda = llambda.at[d, idx].set(llambda_d)
        nu = nu.at[idx, d].set(nu_d)
        nu = nu.at[d, idx].set(nu_d)

        # block inverse update for Q = beta^{-1}
        v = Q_dd_inv @ alpha
        Q_minor = Q_dd_inv + jnp.outer(v, v) / gamma
        Q_cross = -v / gamma

        Q = Q.at[jnp.ix_(idx, idx)].set(Q_minor)
        Q = Q.at[idx, d].set(Q_cross)
        Q = Q.at[d, idx].set(Q_cross)
        Q = Q.at[d, d].set(1 / gamma)

        return beta, Q, llambda, nu

    @classmethod
    def sample(
            cls,
            key,
            N: int,
            scatter: Array,
            beta: Array,
            Q: Array,
            llambda: Array,
            nu: Array,
            tau: Array,
            xi: Array,
        ):
        """
        Perform one graphical-horseshoe Gibbs sweep.

        Parameters
        ----------
        key:      JAX PRNG key.
        N:        Number of observations.
        scatter:  Scatter matrix sum_n y_n y_n.T, shape (D, D).
        beta:     Current precision matrix, beta = Q^{-1}, shape (D, D).
        Q:        Current covariance matrix, shape (D, D).
        llambda:  Matrix containing lambda_ij^2, shape (D, D).
        nu:       Matrix of local auxiliary variables, shape (D, D).
        tau:      Current global shrinkage scale.
        xi:       Current global auxiliary variable.

        Returns
        -------
        Q:        Updated covariance matrix.
        beta:     Updated precision matrix.
        llambda:  Updated local squared shrinkage scales.
        nu:       Updated local auxiliary variables.
        tau:      Updated global shrinkage scale.
        xi:       Updated global auxiliary variable.
        """
        D = scatter.shape[-1]
        keys = jr.split(key, D + 2)

        for d in range(D):
            beta, Q, llambda, nu = cls._sample_column(
                keys[d], d, N, scatter, beta, Q, llambda, nu, tau
            )

        rows, cols = jnp.triu_indices(D, k=1)
        K = D * (D - 1) // 2

        tau_sq = inverse_gamma(
            keys[-2],
            (K + 1) / 2,
            1 / xi + jnp.sum(beta[rows, cols]**2 / (2 * llambda[rows, cols])),
        )
        xi = inverse_gamma(keys[-1], 1, 1 + 1 / tau_sq)

        tau = jnp.sqrt(tau_sq)
        Q = 0.5 * (Q + Q.T)
        beta = 0.5 * (beta + beta.T)

        return Q, beta, llambda, nu, tau, xi


# @classmethod
# def prior(cls, key: PRNGKey, D: int):
#     """
#     Initialise the graphical-horseshoe state from its shrinkage hierarchy.

#     The graphical-horseshoe prior is improper in the diagonal precision
#     elements, so these cannot be sampled from a marginal prior. Instead, the
#     off-diagonal elements and shrinkage variables are sampled from their proper
#     priors, then the diagonal is chosen to make beta positive definite.
#     """
#     local_key, tau_key, xi_key, beta_key = jr.split(key, 4)
#     rows, cols = jnp.triu_indices(D, k=1)
#     Ks = rows.shape[0]

#     # lambda_ij^2 | nu_ij ~ InvGamma(1/2, 1/nu_ij)
#     # nu_ij ~ InvGamma(1/2, 1)
#     nu_keys = jr.split(local_key, Ks)
#     nus = vmap(lambda k: inverse_gamma(k, 0.5, 1.0))(nu_keys)

#     llambda_keys = jr.split(jr.fold_in(local_key, 1), Ks)
#     llambda_values = vmap(lambda k, nu_ij: inverse_gamma(k, 0.5, 1.0 / nu_ij))(llambda_keys, nus)

#     # tau^2 | xi ~ InvGamma(1/2, 1/xi), xi ~ InvGamma(1/2, 1)
#     xi = inverse_gamma(xi_key, 0.5, 1.0)
#     tau_sq = inverse_gamma(tau_key, 0.5, 1.0 / xi)
#     tau = jnp.sqrt(tau_sq)

#     # beta_ij | lambda_ij, tau ~ N(0, lambda_ij^2 tau^2)
#     beta_values = jr.normal(beta_key, shape=(Ks,))
#     beta_values *= jnp.sqrt(llambda_values) * tau

#     # store local variables symmetrically; diagonal entries are unused
#     llambda = jnp.ones((D, D))
#     llambda = llambda.at[rows, cols].set(llambda_values)
#     llambda = llambda.at[cols, rows].set(llambda_values)

#     nu = jnp.ones((D, D))
#     nu = nu.at[rows, cols].set(nus)
#     nu = nu.at[cols, rows].set(nus)

#     # construct a symmetric, strictly diagonally dominant precision matrix
#     beta = jnp.zeros((D, D))
#     beta = beta.at[rows, cols].set(beta_values)
#     beta = beta.at[cols, rows].set(beta_values)
#     beta = beta.at[jnp.diag_indices(D)].set(1.0 + jnp.sum(jnp.abs(beta), axis=1))

#     Q = jnp.linalg.solve(beta, jnp.eye(D))
#     Q = 0.5 * (Q + Q.T)

#     return Q, beta, llambda, nu, tau, xi

